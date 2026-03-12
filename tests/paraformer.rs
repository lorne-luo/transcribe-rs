mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::paraformer::ParaformerModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_paraformer_transcribe() {
    env_logger::init();

    let model_path = PathBuf::from("models/paraformer-large-zh");
    let wav_path = PathBuf::from("samples/test.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return;
    }

    let mut model =
        ParaformerModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    let result = model
        .transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);
}