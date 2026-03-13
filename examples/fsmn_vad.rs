//! FSMN-VAD example: Detect speech segments in audio.
//!
//! Usage:
//!   cargo run --features onnx --example fsmn_vad <model_dir> <wav_file>
//!
//! Example:
//!   cargo run --features onnx --example fsmn_vad \
//!     ~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx \
//!     samples/test.wav

use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::fsmn_vad::{FsmnVadModel, VadConfig};
use transcribe_rs::onnx::Quantization;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let int8 = args.iter().any(|a| a == "--int8");

    let positional: Vec<&String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();

    let model_path = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or("~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx"),
    );
    let wav_path = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("samples/test.wav"),
    );

    let quantization = if int8 {
        Quantization::Int8
    } else {
        Quantization::FP32
    };

    println!("FSMN-VAD: Voice Activity Detection");
    println!("Model: {:?}", model_path);
    println!("Audio: {:?}", wav_path);

    let load_start = Instant::now();
    let config = VadConfig {
        speech_noise_thres: 0.5,
        max_end_silence_ms: 800,
        min_speech_ms: 150,
        window_ms: 200,
    };
    let mut model = FsmnVadModel::load_with_config(&model_path, &quantization, config)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    let samples = read_wav_samples(&wav_path)?;
    let duration = samples.len() as f64 / 16000.0;
    println!("Audio duration: {:.2}s", duration);

    let detect_start = Instant::now();
    let result = model.detect(&samples)?;
    let detect_duration = detect_start.elapsed();

    println!("Detection completed in {:.2?}", detect_duration);
    println!("Found {} speech segments:", result.segments.len());

    for (i, seg) in result.segments.iter().enumerate() {
        println!(
            "  [{}] {}ms - {}ms ({:.2}s - {:.2}s)",
            i + 1,
            seg.start_ms,
            seg.end_ms,
            seg.start_ms as f64 / 1000.0,
            seg.end_ms as f64 / 1000.0
        );
    }

    // Calculate total speech time
    let total_speech: i64 = result.segments.iter().map(|s| s.end_ms - s.start_ms).sum();
    let speech_ratio = total_speech as f64 / (duration * 1000.0);
    println!(
        "\nTotal speech: {:.2}s ({:.1}% of audio)",
        total_speech as f64 / 1000.0,
        speech_ratio * 100.0
    );

    Ok(())
}