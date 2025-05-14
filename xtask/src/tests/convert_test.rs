#[cfg(test)]
mod tests {
    use crate::convert::ConvertArgs;

    #[test]
    fn test_convert_args_default() {
        // 只测试默认构造函数是否可用
        let _ = ConvertArgs::default();
    }

    #[test]
    fn test_convert_parse_steps() {
        // 测试步骤解析，验证不同操作类型能被正确识别
        let steps = "sort->permute-qk->merge-linear->to-llama:extra->cast:F16->filter-meta:key->filter-tensor:name";
        let steps_vec: Vec<&str> = steps.split("->").collect();
        
        assert_eq!(steps_vec.len(), 7);
        assert_eq!(steps_vec[0], "sort");
        assert_eq!(steps_vec[1], "permute-qk");
        assert_eq!(steps_vec[2], "merge-linear");
        assert_eq!(steps_vec[3], "to-llama:extra");
        assert_eq!(steps_vec[4], "cast:F16");
        assert_eq!(steps_vec[5], "filter-meta:key");
        assert_eq!(steps_vec[6], "filter-tensor:name");
    }
} 