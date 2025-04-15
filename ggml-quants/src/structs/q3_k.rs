use super::{_256, f16};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct Q3K {
    pub hmask: [u8; _256 / 8],
    pub qs: [u8; _256 / 4],
    pub scales: [u8; 12],
    pub delta: f16,
}

impl_data_block! {
    Q3K = crate::types::Q3K;
    Self {
        hmask: [0; _256 / 8],
        qs: [0; _256 / 4],
        scales: [0; 12],
        delta: f16::ZERO,
    }
}

impl Quantize<f32, _256> for Q3K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
