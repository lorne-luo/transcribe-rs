//! CT-Transformer punctuation example: Add punctuation to Chinese text.
//!
//! Usage:
//!   cargo run --features onnx --example ct_punc <model_dir> <text>
//!
//! Example:
//!   cargo run --features onnx --example ct_punc \
//!     ~/.cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx \
//!     "你好这是一个测试"

use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::ct_punc::CtPuncModel;
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
            .unwrap_or("~/.cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"),
    );

    let text = positional
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("你好这是一个测试");

    let quantization = if int8 {
        Quantization::Int8
    } else {
        Quantization::FP32
    };

    println!("CT-Transformer: Punctuation Restoration");
    println!("Model: {:?}", model_path);

    let load_start = Instant::now();
    let mut model = CtPuncModel::load(&model_path, &quantization)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    println!("\nInput text: {}", text);

    let infer_start = Instant::now();
    let result = model.punctuate(text)?;
    let infer_duration = infer_start.elapsed();

    println!("Output text: {}", result.text);
    println!("\nInference completed in {:.2?}", infer_duration);

    Ok(())
}