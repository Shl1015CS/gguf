#[cfg(test)]
mod tests {
    use crate::merge::MergeArgs;

    #[test]
    fn test_merge_args_default() {
        // 只测试默认构造函数是否可用
        let _ = MergeArgs::default();
    }
} 