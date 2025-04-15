use super::{_256, f16};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ2XXS {
    pub delta: f16,
    pub qs: [u16; _256 / 8],
}

impl_data_block! {
    IQ2XXS = crate::types::IQ2XXS;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 8],
    }
}

impl Quantize<f32, _256> for IQ2XXS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
