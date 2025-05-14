#[cfg(test)]
mod tests {
    use crate::split::SplitArgs;

    #[test]
    fn test_split_args_default() {
        // 只测试默认构造函数是否可用
        let _ = SplitArgs::default();
    }
} 