use std::fmt;

#[derive(Clone, PartialEq, Debug)]
#[repr(u8)]
pub enum Type {
    Default,
    LoRA,
    Vocab,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Default => Ok(()),
            Type::LoRA => write!(f, "-LoRA"),
            Type::Vocab => write!(f, "-Vocab"),
        }
    }
}
