use super::{_256, f16};
use crate::{DataBlock, Quantize};

#[repr(C)]
pub struct IQ4XS {
    pub delta: f16,
    pub scales_h: u16,
    pub scales_l: [u8; _256 / 64],
    pub qs: [u16; _256 / 2],
}

impl_data_block! {
    IQ4XS = crate::types::IQ4XS;
    Self {
        delta: f16::ZERO,
        scales_h: 0,
        scales_l: [0; _256 / 64],
        qs: [0; _256 / 2],
    }
}

impl Quantize<f32, _256> for IQ4XS {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        todo!()
    }
}
