use super::{_256, f16};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ2S {
    pub delta: f16,
    pub qs: [u8; _256 / 4],
    pub qh: [u8; _256 / 32],
    pub scales: [u8; _256 / 32],
}

impl_data_block! {
    IQ2S = crate::types::IQ2S;
    Self {
        delta: f16::ZERO,
        qs: [0; _256 / 4],
        qh: [0; _256 / 32],
        scales: [0; _256 / 32],
    }
}

impl Quantize<f32, _256> for IQ2S {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
