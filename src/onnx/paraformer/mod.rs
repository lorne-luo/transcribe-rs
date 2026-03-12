//! Paraformer-large ASR model support.
//!
//! A non-autoregressive speech recognition model optimized for Chinese.
//! Model: `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`

use ndarray::Array1;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::fs;
use std::path::Path;

use crate::features::{apply_cmvn, apply_lfr, compute_mel, MelConfig, WindowType};
use crate::TranscribeError;
use super::session;
use super::Quantization;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Paraformer",
    engine_id: "paraformer",
    sample_rate: 16000,
    languages: &["zh"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Per-model inference parameters for Paraformer.
#[derive(Debug, Clone, Default)]
pub struct ParaformerParams {
    /// Language for transcription (ignored, model is Chinese-only).
    pub language: Option<String>,
}

/// Paraformer-large ASR model.
pub struct ParaformerModel {
    session: Session,
    neg_mean: Array1<f32>,
    inv_stddev: Array1<f32>,
    vocab: Vec<String>,
}

impl ParaformerModel {
    /// Load a Paraformer model from a directory.
    ///
    /// Expected files:
    /// - `model.int8.onnx` or `model.onnx`: ONNX model
    /// - `am.mvn`: CMVN statistics (Kaldi Nnet format)
    /// - `tokens.json`: Vocabulary (JSON array of 8404 strings)
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let model_path = session::resolve_model_path(model_dir, "model", quantization);
        let am_mvn_path = model_dir.join("am.mvn");
        let tokens_path = model_dir.join("tokens.json");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !am_mvn_path.exists() {
            return Err(TranscribeError::ModelNotFound(am_mvn_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading Paraformer model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        // Parse CMVN from am.mvn file
        let (neg_mean, inv_stddev) = Self::parse_am_mvn(&am_mvn_path)?;
        log::info!(
            "Loaded CMVN: neg_mean[{}], inv_stddev[{}]",
            neg_mean.len(),
            inv_stddev.len()
        );

        // Load vocabulary from tokens.json
        let vocab = Self::load_tokens_json(&tokens_path)?;
        log::info!("Loaded vocabulary: {} tokens", vocab.len());

        Ok(Self {
            session,
            neg_mean,
            inv_stddev,
            vocab,
        })
    }

    /// Parse Kaldi Nnet format am.mvn file.
    ///
    /// Extracts neg_mean from `<AddShift>` and inv_stddev from `<Rescale>`.
    /// Both are 560-dimensional (80 mel × 7 LFR window).
    fn parse_am_mvn(path: &Path) -> Result<(Array1<f32>, Array1<f32>), TranscribeError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TranscribeError::Config(format!("Failed to read am.mvn: {}", e)))?;

        // Find neg_mean in <AddShift> section
        let neg_mean = Self::extract_bracketed_floats(
            &content,
            "<AddShift>",
            "<LearnRateCoef>",
        )?;

        // Find inv_stddev in <Rescale> section
        let inv_stddev = Self::extract_bracketed_floats(
            &content,
            "<Rescale>",
            "<LearnRateCoef>",
        )?;

        Ok((Array1::from_vec(neg_mean), Array1::from_vec(inv_stddev)))
    }

    /// Extract float array from bracketed section after a header tag.
    fn extract_bracketed_floats(
        content: &str,
        section_header: &str,
        array_header: &str,
    ) -> Result<Vec<f32>, TranscribeError> {
        // Find section
        let section_start = content.find(section_header)
            .ok_or_else(|| TranscribeError::Config(
                format!("Section '{}' not found in am.mvn", section_header)
            ))?;

        let section = &content[section_start..];

        // Find array header after section
        let array_start = section.find(array_header)
            .ok_or_else(|| TranscribeError::Config(
                format!("Array header '{}' not found after {}", array_header, section_header)
            ))?;

        let after_header = &section[array_start + array_header.len()..];

        // Find opening bracket
        let bracket_start = after_header.find('[')
            .ok_or_else(|| TranscribeError::Config("Opening bracket not found".into()))?;

        let after_open = &after_header[bracket_start + 1..];

        // Find closing bracket
        let bracket_end = after_open.find(']')
            .ok_or_else(|| TranscribeError::Config("Closing bracket not found".into()))?;

        let numbers_str = &after_open[..bracket_end];

        // Parse floats
        let floats: Result<Vec<f32>, _> = numbers_str
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect();

        floats.map_err(|e| TranscribeError::Config(format!("Failed to parse floats: {}", e)))
    }

    /// Load tokens.json vocabulary file.
    ///
    /// Format: JSON array of strings (8404 tokens).
    fn load_tokens_json(path: &Path) -> Result<Vec<String>, TranscribeError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TranscribeError::Config(format!("Failed to read tokens.json: {}", e)))?;

        let vocab: Vec<String> = serde_json::from_str(&content)
            .map_err(|e| TranscribeError::Config(format!("Failed to parse tokens.json: {}", e)))?;

        Ok(vocab)
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &ParaformerParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        // Copy values to avoid borrow conflicts
        let neg_mean = self.neg_mean.clone();
        let inv_stddev = self.inv_stddev.clone();

        // 1. Compute FBANK features (80 mel bins)
        let mel_config = MelConfig {
            sample_rate: 16000,
            num_mels: 80,
            n_fft: 400,
            hop_length: 160,
            window: WindowType::Hamming,
            f_min: 20.0,
            f_max: None,
            pre_emphasis: Some(0.97),
            snip_edges: true,
            normalize_samples: false,
        };
        let features = compute_mel(samples, &mel_config);

        log::debug!(
            "FBANK features: [{}, {}]",
            features.nrows(),
            features.ncols()
        );

        // 2. Apply LFR (window=7, shift=6) -> 560-dim features
        let features = apply_lfr(&features, 7, 6);
        log::debug!("After LFR: [{}, {}]", features.nrows(), features.ncols());

        if features.nrows() == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // 3. Apply CMVN
        let mut features = features;
        apply_cmvn(&mut features, &neg_mean, &inv_stddev);

        // 4. Run ONNX forward pass
        let (logits, token_num) = self.forward(&features.view())?;

        log::debug!("Logits shape: {:?}, token_num: {}", logits.shape(), token_num);

        // 5. Argmax decode (non-autoregressive)
        let text = self.argmax_decode(&logits, token_num as usize);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    fn forward(
        &mut self,
        features: &ndarray::ArrayView2<f32>,
    ) -> Result<(ndarray::Array3<f32>, i32), TranscribeError> {
        let num_frames = features.nrows() as i32;

        // Reshape to [batch=1, T, 560]
        let feat_3d = features
            .to_owned()
            .into_shape_with_order((1, features.nrows(), features.ncols()))?;

        let speech_lengths = ndarray::arr1(&[num_frames]);

        let feat_dyn = feat_3d.into_dyn();
        let len_dyn = speech_lengths.into_dyn();

        let t_feat = TensorRef::from_array_view(feat_dyn.view())?;
        let t_len = TensorRef::from_array_view(len_dyn.view())?;

        // ONNX inputs: speech, speech_lengths
        let inputs = inputs![
            "speech" => t_feat,
            "speech_lengths" => t_len,
        ];

        let outputs = self.session.run(inputs)?;

        // ONNX outputs: logits [batch, N, vocab], token_num [batch]
        let logits = outputs[0].try_extract_array::<f32>()?;
        let logits_owned = logits.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        let token_num_arr = outputs[1].try_extract_array::<i32>()?;
        let token_num = token_num_arr[[0]];

        Ok((logits_owned, token_num))
    }

    /// Argmax decode for non-autoregressive Paraformer.
    ///
    /// For each time step, take argmax over vocabulary dimension.
    /// Skip special tokens: <blank>=0, <s>=1, </s>=2, <unk>=8403.
    /// Strip `@@` BPE continuation markers.
    fn argmax_decode(&self, logits: &ndarray::Array3<f32>, token_num: usize) -> String {
        let mut tokens = Vec::new();

        // logits shape: [1, N, vocab_size]
        for t in 0..token_num.min(logits.shape()[1]) {
            let frame_logits = logits.slice(ndarray::s![0, t, ..]);

            // Argmax
            let mut max_idx = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in frame_logits.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }

            // Skip special tokens
            if max_idx == 0 || max_idx == 1 || max_idx == 2 || max_idx == 8403 {
                continue;
            }

            // Get token string
            if let Some(token) = self.vocab.get(max_idx) {
                tokens.push(token.clone());
            }
        }

        // Join tokens, stripping @@ BPE markers
        let mut text = String::new();
        for token in tokens {
            let token = token.trim_end_matches("@@");
            text.push_str(&token);
        }

        text
    }
}

impl SpeechModel for ParaformerModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}