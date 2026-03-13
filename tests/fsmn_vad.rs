mod common;

use std::path::PathBuf;

use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::fsmn_vad::{FsmnVadModel, VadConfig};
use transcribe_rs::onnx::Quantization;

fn get_model_path() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx"))
        .expect("No home directory")
}

#[test]
fn test_fsmn_vad_detect_speech() {
    let model_path = get_model_path();
    let wav_path = PathBuf::from("samples/test.wav");

    common::require_paths_or_panic(&[&model_path, &wav_path]);

    let config = VadConfig::default();
    let mut model = FsmnVadModel::load_with_config(&model_path, &Quantization::Int8, config)
        .expect("Failed to load VAD model");

    let samples = read_wav_samples(&wav_path).expect("Failed to read WAV file");
    let duration = samples.len() as f64 / 16000.0;

    // Expected: 7.92s audio duration
    assert!((duration - 7.92).abs() < 0.01, "Audio duration should be ~7.92s, got {:.2}s", duration);

    let result = model.detect(&samples).expect("Failed to detect speech");

    println!("Audio duration: {:.2}s", duration);
    println!("Found {} speech segments", result.segments.len());

    for (i, seg) in result.segments.iter().enumerate() {
        println!(
            "  [{}] {}ms - {}ms",
            i + 1,
            seg.start_ms,
            seg.end_ms
        );
    }

    // Expected: 1 segment (20ms-7850ms)
    assert_eq!(result.segments.len(), 1, "Should detect exactly 1 speech segment");

    let seg = &result.segments[0];
    // Allow some tolerance for VAD detection
    assert!(seg.start_ms <= 50, "Segment start should be near 20ms, got {}ms", seg.start_ms);
    assert!(seg.end_ms >= 7800, "Segment end should be near 7850ms, got {}ms", seg.end_ms);

    // Total speech: 7.83s (98.8% of audio)
    let total_speech: i64 = result.segments.iter().map(|s| s.end_ms - s.start_ms).sum();
    let speech_ratio = total_speech as f64 / (duration * 1000.0);
    assert!(speech_ratio > 0.95, "Speech ratio should be >95%, got {:.1}%", speech_ratio * 100.0);
}

#[test]
fn test_fsmn_vad_config() {
    let config = VadConfig {
        speech_noise_thres: 0.5,
        max_end_silence_ms: 1000,
        min_speech_ms: 200,
        window_ms: 200,
    };

    assert_eq!(config.speech_noise_thres, 0.5);
    assert_eq!(config.max_end_silence_ms, 1000);
    assert_eq!(config.min_speech_ms, 200);
    assert_eq!(config.window_ms, 200);
}

#[test]
fn test_fsmn_vad_default_config() {
    let config = VadConfig::default();

    // Default values
    assert_eq!(config.speech_noise_thres, 0.5);
    assert_eq!(config.max_end_silence_ms, 800);
    assert_eq!(config.min_speech_ms, 150);
    assert_eq!(config.window_ms, 200);
}