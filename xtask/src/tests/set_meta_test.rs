#[cfg(test)]
mod tests {
    use crate::set_meta::SetMetaArgs;

    #[test]
    fn test_set_meta_args_default() {
        // 只测试默认构造函数是否可用
        let _ = SetMetaArgs::default();
    }
} 