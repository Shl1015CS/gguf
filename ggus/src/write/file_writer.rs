use super::GGufWriter;
use crate::{DEFAULT_ALIGNMENT, GGmlType, GGufFileHeader, GGufMetaDataValueType, pad};
use log::trace;
use std::{
    borrow::Borrow,
    io::{Result, Write},
    time::Instant,
};

pub struct GGufFileWriter<T: Write> {
    writer: GGufWriter<T>,
    alignment: usize,
}

pub struct GGufTensorWriter<T: Write, U> {
    writer: GGufWriter<T>,
    alignment: usize,
    data: Vec<U>,
    offset: usize,
    write_data: bool,
}

pub trait DataFuture {
    fn get(&self) -> &[u8];
}

impl<T: Borrow<[u8]>> DataFuture for T {
    #[inline]
    fn get(&self) -> &[u8] {
        self.borrow()
    }
}

impl<T: Write> GGufFileWriter<T> {
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        let mut writer = GGufWriter::new(writer);
        writer.write_header(header)?;
        Ok(Self {
            writer,
            alignment: DEFAULT_ALIGNMENT,
        })
    }

    #[inline]
    pub fn with_alignment(writer: T, header: GGufFileHeader, alignment: usize) -> Result<Self> {
        let mut ans = Self::new(writer, header)?;
        ans.write_alignment(alignment)?;
        Ok(ans)
    }

    #[inline]
    pub fn write_alignment(&mut self, alignment: usize) -> Result<()> {
        self.writer.write_alignment(alignment)?;
        self.alignment = alignment;
        Ok(())
    }

    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: &str,
        ty: GGufMetaDataValueType,
        val: &[u8],
    ) -> Result<()> {
        if let Some(alignment) = self.writer.write_meta_kv(key, ty, val)? {
            self.alignment = alignment;
        }
        Ok(())
    }

    #[inline]
    pub fn finish<U>(self, write_data: bool) -> GGufTensorWriter<T, U> {
        GGufTensorWriter {
            writer: self.writer,
            alignment: self.alignment,
            data: Vec::new(),
            offset: 0,
            write_data,
        }
    }
}

impl<T: Write, U: DataFuture> GGufTensorWriter<T, U> {
    pub fn write_tensor(&mut self, name: &str, ty: GGmlType, shape: &[u64], data: U) -> Result<()> {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(name, shape, ty, self.offset as _)
            .unwrap();

        let len = ty.size().elements_to_bytes(shape);
        self.offset += len;
        if self.write_data {
            self.data.push(data)
        }
        Ok(())
    }

    pub fn finish(self) -> Result<usize> {
        let Self {
            mut writer,
            alignment,
            data,
            ..
        } = self;

        let total = data.len().to_string();
        let width = total.len();
        for (i, data) in data.into_iter().enumerate() {
            let t0 = Instant::now();
            let data = data.get();
            let t1 = Instant::now();
            writer.write_padding(alignment)?;
            writer.write_data(data)?;
            let t2 = Instant::now();
            trace!(
                "data {i:>width$}/{total} size = {} bytes, calculate in {:?}, write in {:?}",
                data.len(),
                t1 - t0,
                t2 - t1,
            )
        }
        Ok(writer.written_bytes())
    }
}
