#[cfg(test)]
mod tests {
    use crate::*;
    use clap::Parser;

    #[test]
    fn test_command_routing() {
        // 测试命令行路由到正确的处理函数
        let args = vec![
            "gguf-utils", 
            "show", 
            "non_existent_file.gguf", 
            "--array-detail", "all"
        ];
        
        let cli = Cli::parse_from(args);
        
        match cli.command {
            Commands::Show(_) => {
                // 验证命令类型正确
                assert!(true);
            },
            _ => panic!("Wrong command parsed"),
        }
    }
} 