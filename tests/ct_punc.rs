mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::ct_punc::CtPuncModel;
use transcribe_rs::onnx::Quantization;

fn get_model_path() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".cache/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx"))
        .expect("No home directory")
}

#[test]
fn test_ct_punc_punctuate() {
    let model_path = get_model_path();

    common::require_paths_or_panic(&[&model_path]);

    let mut model = CtPuncModel::load(&model_path, &Quantization::Int8)
        .expect("Failed to load punctuation model");

    // Test case from actual run:
    // Input: "你好这是一个测试"
    // Output: "你好，这是一个测试。"
    let result = model.punctuate("你好这是一个测试").expect("Failed to punctuate");
    println!("Input: 你好这是一个测试 -> Output: {}", result.text);

    // Verify the expected output
    assert!(result.text.contains("，"), "Should contain comma '，'");
    assert!(result.text.contains("。"), "Should contain period '。'");

    // Verify structure: should start with "你好" and end with "。"
    assert!(result.text.starts_with("你好"), "Should start with '你好'");
    assert!(result.text.ends_with("。"), "Should end with '。'");
}

#[test]
fn test_ct_punc_multiple_cases() {
    let model_path = get_model_path();

    common::require_paths_or_panic(&[&model_path]);

    let mut model = CtPuncModel::load(&model_path, &Quantization::Int8)
        .expect("Failed to load punctuation model");

    let test_cases = vec![
        // (input, should_have_punctuation)
        ("今天天气很好", true),
        ("我是中国人", true),
        ("这是一个测试", true),
    ];

    for (input, should_have_punc) in test_cases {
        let result = model.punctuate(input).expect("Failed to punctuate");
        println!("Input: {} -> Output: {}", input, result.text);

        if should_have_punc {
            // Output should contain some form of punctuation
            let has_punc = result.text.contains('，') ||
                           result.text.contains('。') ||
                           result.text.contains('？') ||
                           result.text.contains('、');
            assert!(has_punc || result.text.len() > input.len(),
                    "Should add punctuation to '{}'", input);
        }
    }
}

#[test]
fn test_ct_punc_empty_input() {
    let model_path = get_model_path();

    common::require_paths_or_panic(&[&model_path]);

    let mut model = CtPuncModel::load(&model_path, &Quantization::Int8)
        .expect("Failed to load punctuation model");

    let result = model.punctuate("").expect("Failed to punctuate empty string");
    assert!(result.text.is_empty(), "Empty input should produce empty output");
}

#[test]
fn test_ct_punc_preserves_characters() {
    let model_path = get_model_path();

    common::require_paths_or_panic(&[&model_path]);

    let mut model = CtPuncModel::load(&model_path, &Quantization::Int8)
        .expect("Failed to load punctuation model");

    let input = "你好这是一个测试";
    let result = model.punctuate(input).expect("Failed to punctuate");

    // All original characters should be preserved (only punctuation added)
    for ch in input.chars() {
        assert!(
            result.text.contains(ch),
            "Character '{}' should be preserved in output '{}'",
            ch,
            result.text
        );
    }

    // Output should be longer than input (punctuation added)
    assert!(result.text.len() > input.len(),
            "Output '{}' should be longer than input '{}'",
            result.text, input);
}