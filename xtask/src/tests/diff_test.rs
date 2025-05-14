#[cfg(test)]
mod tests {
    use crate::diff::DiffArgs;

    #[test]
    fn test_diff_args_default() {
        // 只测试默认构造函数是否可用
        let _ = DiffArgs::default();
    }
} 