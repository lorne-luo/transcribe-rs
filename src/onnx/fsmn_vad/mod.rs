//! FSMN-VAD (Voice Activity Detection) model support.
//!
//! A streaming VAD model for detecting speech segments in audio.
//! Model: `speech_fsmn_vad_zh-cn-16k-common-onnx`
//!
//! # Model Architecture
//!
//! - Frontend: WavFrontendOnline (80 mel, LFR m=5, n=1)
//! - Encoder: 4 FSMN layers with left context only
//! - Input: [batch, T, 400] FBANK features
//! - Output: [batch, T, 248] speech probability vectors
//!
//! # Usage
//!
//! ```ignore
//! use transcribe_rs::onnx::fsmn_vad::FsmnVadModel;
//! use transcribe_rs::onnx::Quantization;
//!
//! let model = FsmnVadModel::load(&model_dir, &Quantization::Int8)?;
//! let segments = model.detect(&samples)?;
//! for seg in &segments {
//!     println!("Speech: {}ms - {}ms", seg.start_ms, seg.end_ms);
//! }
//! ```

use ndarray::{Array1, Array4};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::fs;
use std::path::Path;

use crate::features::{apply_cmvn, compute_mel, MelConfig, WindowType};
use crate::TranscribeError;
use super::session;
use super::Quantization;

/// VAD detection thresholds.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Threshold for speech detection (default: 0.5)
    pub speech_noise_thres: f32,
    /// Maximum silence duration in ms before ending segment (default: 800)
    pub max_end_silence_ms: i64,
    /// Minimum speech duration in ms (default: 150)
    pub min_speech_ms: i64,
    /// Window size in ms for smoothing (default: 200)
    pub window_ms: i64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_noise_thres: 0.5,
            max_end_silence_ms: 800,
            min_speech_ms: 150,
            window_ms: 200,
        }
    }
}

/// A detected speech segment.
#[derive(Debug, Clone, Copy)]
pub struct VadSegment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
}

/// Result of VAD detection.
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Detected speech segments.
    pub segments: Vec<VadSegment>,
}

/// FSMN-VAD model for voice activity detection.
pub struct FsmnVadModel {
    session: Session,
    neg_mean: Array1<f32>,
    inv_stddev: Array1<f32>,
    config: VadConfig,
    /// Frame duration in ms (10ms per frame for 16kHz with hop=160)
    frame_ms: i64,
}

impl FsmnVadModel {
    /// Load a FSMN-VAD model from a directory.
    ///
    /// Expected files:
    /// - `model_quant.onnx` or `model.onnx`: ONNX model
    /// - `am.mvn`: CMVN statistics
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        Self::load_with_config(model_dir, quantization, VadConfig::default())
    }

    /// Load with custom VAD configuration.
    pub fn load_with_config(
        model_dir: &Path,
        quantization: &Quantization,
        config: VadConfig,
    ) -> Result<Self, TranscribeError> {
        let model_path = session::resolve_model_path(model_dir, "model", quantization);
        let am_mvn_path = model_dir.join("am.mvn");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !am_mvn_path.exists() {
            return Err(TranscribeError::ModelNotFound(am_mvn_path));
        }

        log::info!("Loading FSMN-VAD model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        // Parse CMVN from am.mvn file
        let (neg_mean, inv_stddev) = Self::parse_am_mvn(&am_mvn_path)?;
        log::info!(
            "Loaded CMVN: neg_mean[{}], inv_stddev[{}]",
            neg_mean.len(),
            inv_stddev.len()
        );

        Ok(Self {
            session,
            neg_mean,
            inv_stddev,
            config,
            frame_ms: 10, // 10ms per frame (hop=160 at 16kHz)
        })
    }

    /// Parse Kaldi Nnet format am.mvn file (same as Paraformer).
    fn parse_am_mvn(path: &Path) -> Result<(Array1<f32>, Array1<f32>), TranscribeError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TranscribeError::Config(format!("Failed to read am.mvn: {}", e)))?;

        let neg_mean = Self::extract_bracketed_floats(&content, "<AddShift>", "<LearnRateCoef>")?;
        let inv_stddev = Self::extract_bracketed_floats(&content, "<Rescale>", "<LearnRateCoef>")?;

        Ok((Array1::from_vec(neg_mean), Array1::from_vec(inv_stddev)))
    }

    fn extract_bracketed_floats(
        content: &str,
        section_header: &str,
        array_header: &str,
    ) -> Result<Vec<f32>, TranscribeError> {
        let section_start = content.find(section_header).ok_or_else(|| {
            TranscribeError::Config(format!("Section '{}' not found in am.mvn", section_header))
        })?;

        let section = &content[section_start..];
        let array_start = section.find(array_header).ok_or_else(|| {
            TranscribeError::Config(format!(
                "Array header '{}' not found after {}",
                array_header, section_header
            ))
        })?;

        let after_header = &section[array_start + array_header.len()..];
        let bracket_start = after_header
            .find('[')
            .ok_or_else(|| TranscribeError::Config("Opening bracket not found".into()))?;

        let after_open = &after_header[bracket_start + 1..];
        let bracket_end = after_open
            .find(']')
            .ok_or_else(|| TranscribeError::Config("Closing bracket not found".into()))?;

        let numbers_str = &after_open[..bracket_end];
        let floats: Result<Vec<f32>, _> = numbers_str
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect();

        floats.map_err(|e| TranscribeError::Config(format!("Failed to parse floats: {}", e)))
    }

    /// Detect speech segments in audio samples.
    ///
    /// Input: 16kHz mono audio samples (f32 in [-1, 1])
    /// Output: List of speech segments [start_ms, end_ms]
    pub fn detect(&mut self, samples: &[f32]) -> Result<VadResult, TranscribeError> {
        // 1. Compute FBANK features (80 mel bins)
        let mel_config = MelConfig {
            sample_rate: 16000,
            num_mels: 80,
            n_fft: 400,
            hop_length: 160, // 10ms frame shift
            window: WindowType::Hamming,
            f_min: 20.0,
            f_max: None,
            pre_emphasis: Some(0.97),
            snip_edges: true,
            normalize_samples: false,
        };
        let features = compute_mel(samples, &mel_config);

        if features.nrows() == 0 {
            return Ok(VadResult { segments: vec![] });
        }

        log::debug!("FBANK features: [{}, {}]", features.nrows(), features.ncols());

        // 2. Apply LFR (m=5, n=1) -> 400-dim features
        // m=5: stack 5 consecutive frames
        // n=1: shift by 1 frame (no downsampling)
        let features = Self::apply_lfr_online(&features.view(), 5, 1);
        log::debug!("After LFR: [{}, {}]", features.nrows(), features.ncols());

        // 3. Apply CMVN
        let mut features = features;
        apply_cmvn(&mut features, &self.neg_mean, &self.inv_stddev);

        // 4. Run streaming inference
        let probs = self.infer_streaming(&features.view())?;

        // 5. Apply threshold and extract segments
        let segments = self.extract_segments(&probs);

        Ok(VadResult { segments })
    }

    /// Apply LFR for online mode (m=5, n=1).
    ///
    /// Unlike Paraformer's LFR (m=7, n=6 which downsamples),
    /// online VAD uses m=5, n=1 which preserves frame rate.
    fn apply_lfr_online(
        features: &ndarray::ArrayView2<f32>,
        lfr_m: usize,
        lfr_n: usize,
    ) -> ndarray::Array2<f32> {
        let num_frames = features.nrows();
        let num_mels = features.ncols();

        if num_frames < lfr_m {
            // Not enough frames for LFR window
            let mut result = ndarray::Array2::zeros((1, num_mels * lfr_m));
            for (i, row) in features.rows().into_iter().enumerate() {
                let start = i * num_mels;
                result.slice_mut(ndarray::s![0, start..start + num_mels]).assign(&row);
            }
            return result;
        }

        let output_frames = (num_frames - lfr_m) / lfr_n + 1;
        let mut result = ndarray::Array2::zeros((output_frames, num_mels * lfr_m));

        for i in 0..output_frames {
            let start_frame = i * lfr_n;
            for j in 0..lfr_m {
                let frame_idx = start_frame + j;
                let out_start = j * num_mels;
                result
                    .slice_mut(ndarray::s![i, out_start..out_start + num_mels])
                    .assign(&features.slice(ndarray::s![frame_idx, ..]));
            }
        }

        result
    }

    /// Run streaming FSMN inference.
    ///
    /// The FSMN model maintains hidden state (cache) across frames.
    /// We process all frames in one batch for efficiency.
    fn infer_streaming(
        &mut self,
        features: &ndarray::ArrayView2<f32>,
    ) -> Result<Vec<f32>, TranscribeError> {
        let num_frames = features.nrows() as i32;

        // Reshape to [batch=1, T, 400]
        let feat_3d = features
            .to_owned()
            .into_shape_with_order((1, features.nrows(), features.ncols()))?;

        // Initialize cache tensors [1, 128, 19, 1] for each layer
        let cache_shape = (1, 128, 19, 1);
        let cache0 = Array4::<f32>::zeros(cache_shape);
        let cache1 = Array4::<f32>::zeros(cache_shape);
        let cache2 = Array4::<f32>::zeros(cache_shape);
        let cache3 = Array4::<f32>::zeros(cache_shape);

        let feat_dyn = feat_3d.into_dyn();
        let cache0_dyn = cache0.into_dyn();
        let cache1_dyn = cache1.into_dyn();
        let cache2_dyn = cache2.into_dyn();
        let cache3_dyn = cache3.into_dyn();

        let t_feat = TensorRef::from_array_view(feat_dyn.view())?;
        let t_cache0 = TensorRef::from_array_view(cache0_dyn.view())?;
        let t_cache1 = TensorRef::from_array_view(cache1_dyn.view())?;
        let t_cache2 = TensorRef::from_array_view(cache2_dyn.view())?;
        let t_cache3 = TensorRef::from_array_view(cache3_dyn.view())?;

        // ONNX inputs
        let inputs = inputs![
            "speech" => t_feat,
            "in_cache0" => t_cache0,
            "in_cache1" => t_cache1,
            "in_cache2" => t_cache2,
            "in_cache3" => t_cache3,
        ];

        let outputs = self.session.run(inputs)?;

        // Output: logits [1, T, 248]
        let logits = outputs[0].try_extract_array::<f32>()?;

        // Sum over the 248-dim output to get speech probability
        // The model outputs a probability distribution; we take the speech class probability
        let mut probs = Vec::with_capacity(num_frames as usize);
        for t in 0..num_frames as usize {
            let frame_logits = logits.slice(ndarray::s![0, t, ..]);
            // Sum probabilities (assuming speech classes)
            let prob: f32 = frame_logits.sum();
            probs.push(prob);
        }

        // Normalize to [0, 1] using sigmoid-like normalization
        // The model outputs log probabilities, so we apply softmax/sigmoid
        let max_prob = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_prob = probs.iter().cloned().fold(f32::INFINITY, f32::min);
        let range = (max_prob - min_prob).max(1e-6);

        for p in &mut probs {
            *p = (*p - min_prob) / range;
        }

        Ok(probs)
    }

    /// Extract speech segments from frame probabilities.
    fn extract_segments(&self, probs: &[f32]) -> Vec<VadSegment> {
        if probs.is_empty() {
            return vec![];
        }

        let threshold = self.config.speech_noise_thres;
        let min_speech_frames = (self.config.min_speech_ms as f64 / self.frame_ms as f64).ceil() as i64;
        let max_silence_frames = (self.config.max_end_silence_ms as f64 / self.frame_ms as f64).ceil() as i64;

        let mut segments: Vec<VadSegment> = vec![];
        let mut in_speech = false;
        let mut speech_start: i64 = 0;
        let mut silence_count: i64 = 0;

        for (i, &prob) in probs.iter().enumerate() {
            let frame_idx = i as i64;

            if prob >= threshold {
                if !in_speech {
                    // Speech onset
                    in_speech = true;
                    speech_start = frame_idx;
                    silence_count = 0;
                } else {
                    silence_count = 0;
                }
            } else {
                if in_speech {
                    silence_count += 1;
                    if silence_count >= max_silence_frames {
                        // End of speech segment
                        let end_frame = frame_idx - silence_count;
                        let duration_frames = end_frame - speech_start;

                        if duration_frames >= min_speech_frames {
                            segments.push(VadSegment {
                                start_ms: speech_start * self.frame_ms,
                                end_ms: end_frame * self.frame_ms,
                            });
                        }
                        in_speech = false;
                        silence_count = 0;
                    }
                }
            }
        }

        // Handle final segment
        if in_speech {
            let end_frame = probs.len() as i64 - silence_count;
            let duration_frames = end_frame - speech_start;

            if duration_frames >= min_speech_frames {
                segments.push(VadSegment {
                    start_ms: speech_start * self.frame_ms,
                    end_ms: end_frame * self.frame_ms,
                });
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_lfr_online() {
        // Create test features [10, 80]
        let features = ndarray::Array2::from_shape_fn((10, 80), |(i, j)| (i * 80 + j) as f32);

        // Apply LFR m=5, n=1 -> output should be [6, 400]
        let result = FsmnVadModel::apply_lfr_online(&features.view(), 5, 1);

        assert_eq!(result.nrows(), 6); // 10 - 5 + 1 = 6
        assert_eq!(result.ncols(), 400); // 80 * 5
    }
}