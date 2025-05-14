#[cfg(test)]
mod tests {
    use crate::utils::compile_patterns;

    #[test]
    fn test_compile_patterns() {
        // 测试默认模式 "*" (匹配所有)
        let pattern = compile_patterns("*");
        assert!(pattern.is_match("test"));
        assert!(pattern.is_match("abc"));

        // 测试特定模式
        let pattern = compile_patterns("test.*");
        assert!(pattern.is_match("test.abc"));
        assert!(!pattern.is_match("abc"));

        // 测试多个模式用 "|" 分隔
        let pattern = compile_patterns("foo|bar");
        assert!(pattern.is_match("foo"));
        assert!(pattern.is_match("bar"));
        assert!(!pattern.is_match("baz"));
    }
} 