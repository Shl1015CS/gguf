#[cfg(test)]
mod tests {
    use crate::show::ShowArgs;

    #[test]
    fn test_show_args_default() {
        // 只测试默认构造函数是否可用
        let _ = ShowArgs::default();
    }
} 