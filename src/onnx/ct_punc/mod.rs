//! CT-Transformer punctuation restoration model support.
//!
//! A transformer-based model for adding punctuation to Chinese text.
//! Model: `punc_ct-transformer_zh-cn-common-vocab272727-onnx`
//!
//! # Model Architecture
//!
//! - Encoder: SANM (Self-Attention with Neural Masking), 4 blocks, 8 heads
//! - Input: Token IDs [batch, T] (272,727 vocab)
//! - Output: Punctuation class logits [batch, T, 6]
//!
//! # Punctuation Classes
//!
//! | ID | Symbol | Meaning |
//! |----|--------|---------|
//! | 0 | `<unk>` | Unknown |
//! | 1 | `_` | No punctuation |
//! | 2 | `，` | Comma |
//! | 3 | `。` | Period |
//! | 4 | `？` | Question mark |
//! | 5 | `、` | Enumeration comma |
//!
//! # Usage
//!
//! ```ignore
//! use transcribe_rs::onnx::ct_punc::CtPuncModel;
//! use transcribe_rs::onnx::Quantization;
//!
//! let model = CtPuncModel::load(&model_dir, &Quantization::Int8)?;
//! let punctuated = model.punctuate("你好这是一个测试")?;
//! // Result: "你好，这是一个测试。"
//! ```

use ndarray::{Array1, Array2};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::TranscribeError;
use super::session;
use super::Quantization;

/// Punctuation symbols in order of model output indices.
const PUNC_LIST: &[&str] = &["<unk>", "_", "，", "。", "？", "、"];

/// Index of period (。) in punctuation list - used for sentence end.
const SENTENCE_END_ID: usize = 3;

/// Result of punctuation restoration.
#[derive(Debug, Clone)]
pub struct PuncResult {
    /// Text with punctuation inserted.
    pub text: String,
}

/// CT-Transformer punctuation model.
pub struct CtPuncModel {
    session: Session,
    /// Token to ID mapping (272,727 tokens).
    token_to_id: HashMap<String, i32>,
    /// ID to token mapping.
    id_to_token: HashMap<i32, String>,
}

impl CtPuncModel {
    /// Load a CT-Transformer punctuation model from a directory.
    ///
    /// Expected files:
    /// - `model_quant.onnx` or `model.onnx`: ONNX model
    /// - `tokens.json`: Vocabulary (272,727 tokens)
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let model_path = session::resolve_model_path(model_dir, "model", quantization);
        let tokens_path = model_dir.join("tokens.json");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading CT-Transformer punctuation model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        // Load vocabulary
        let (token_to_id, id_to_token) = Self::load_tokens(&tokens_path)?;
        log::info!("Loaded vocabulary: {} tokens", token_to_id.len());

        Ok(Self {
            session,
            token_to_id,
            id_to_token,
        })
    }

    /// Load tokens.json vocabulary file.
    fn load_tokens(path: &Path) -> Result<(HashMap<String, i32>, HashMap<i32, String>), TranscribeError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TranscribeError::Config(format!("Failed to read tokens.json: {}", e)))?;

        let tokens: Vec<String> = serde_json::from_str(&content)
            .map_err(|e| TranscribeError::Config(format!("Failed to parse tokens.json: {}", e)))?;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (id, token) in tokens.into_iter().enumerate() {
            let id = id as i32;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        Ok((token_to_id, id_to_token))
    }

    /// Add punctuation to text.
    ///
    /// Input: Raw Chinese text without punctuation.
    /// Output: Text with punctuation inserted.
    pub fn punctuate(&mut self, text: &str) -> Result<PuncResult, TranscribeError> {
        if text.is_empty() {
            return Ok(PuncResult { text: String::new() });
        }

        // 1. Tokenize input text
        let (token_ids, char_positions) = self.tokenize(text);

        if token_ids.is_empty() {
            return Ok(PuncResult { text: text.to_string() });
        }

        // 2. Run ONNX inference
        let logits = self.infer(&token_ids)?;

        // 3. Decode punctuation and insert into text
        let punctuated = self.decode_punctuation(text, &logits, &char_positions);

        Ok(PuncResult { text: punctuated })
    }

    /// Tokenize input text to token IDs.
    ///
    /// Returns (token_ids, char_positions) where char_positions maps
    /// token index to character index in the original text.
    fn tokenize(&self, text: &str) -> (Vec<i32>, Vec<usize>) {
        let mut token_ids = Vec::new();
        let mut char_positions = Vec::new();

        for (char_idx, ch) in text.char_indices() {
            // Try to find the character in vocabulary
            let token = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&token) {
                token_ids.push(id);
                char_positions.push(char_idx);
            } else {
                // Use <unk> token (id=0) for unknown characters
                token_ids.push(0);
                char_positions.push(char_idx);
            }
        }

        (token_ids, char_positions)
    }

    /// Run ONNX inference.
    fn infer(&mut self, token_ids: &[i32]) -> Result<Array2<f32>, TranscribeError> {
        let num_tokens = token_ids.len() as i32;

        // Input tensors
        let input_array = Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())?;
        let length_array = Array1::from_vec(vec![num_tokens]);

        let input_dyn = input_array.into_dyn();
        let length_dyn = length_array.into_dyn();

        let t_input = TensorRef::from_array_view(input_dyn.view())?;
        let t_length = TensorRef::from_array_view(length_dyn.view())?;

        // ONNX inputs
        let inputs = inputs![
            "inputs" => t_input,
            "text_lengths" => t_length,
        ];

        let outputs = self.session.run(inputs)?;

        // Output: logits [1, T, 6]
        let logits = outputs[0].try_extract_array::<f32>()?;
        let logits_2d = logits
            .slice(ndarray::s![0, .., ..])
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok(logits_2d)
    }

    /// Decode punctuation from logits and insert into text.
    fn decode_punctuation(
        &self,
        text: &str,
        logits: &Array2<f32>,
        _char_positions: &[usize],
    ) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut result = String::new();

        log::debug!("Decoding {} chars, {} logits", chars.len(), logits.nrows());

        for (i, ch) in chars.iter().enumerate() {
            result.push(*ch);

            // Each character position maps to a logit
            // The model outputs logits for positions between characters
            // Position i means punctuation AFTER character i
            if i < logits.nrows() {
                let punc_logits = logits.row(i);
                let mut max_class = 0;
                let mut max_val = f32::NEG_INFINITY;

                for (c, &v) in punc_logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_class = c;
                    }
                }

                log::debug!("Char '{}' at pos {}: class {} ({:?})", ch, i, max_class, PUNC_LIST.get(max_class));

                // Insert punctuation if not "_" or "<unk>"
                if max_class > 1 && max_class < PUNC_LIST.len() {
                    result.push_str(PUNC_LIST[max_class]);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_punc_list() {
        assert_eq!(PUNC_LIST[0], "<unk>");
        assert_eq!(PUNC_LIST[1], "_");
        assert_eq!(PUNC_LIST[2], "，");
        assert_eq!(PUNC_LIST[3], "。");
        assert_eq!(PUNC_LIST[4], "？");
        assert_eq!(PUNC_LIST[5], "、");
    }
}