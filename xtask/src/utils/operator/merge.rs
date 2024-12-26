use super::{super::Tensor, Content, DataPromise};
use ggus::{DataFuture, GGmlType, GGmlTypeSize};
use mem_rearrange::{ndarray_layout::ArrayLayout, Rearranging};
use memmap2::MmapMut;
use regex::Regex;
use std::{borrow::Cow, collections::HashMap, hash::Hash, iter::zip, sync::LazyLock};

const MERGE: &str = r"(attn\.q|attn_q|attn\.k|attn_k|attn\.v|attn_v|ffn_gate|ffn_up|ffn_gate_exps|ffn_up_exps)\.(weight|bias)$";
const SPLIT: &str = r"(attn_qkv|ffn_gate_up)\.(weight|bias)$";
const ATTN_QKV: &str = "attn_qkv";
const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const FFN_GATE_UP: &str = "ffn_gate_up";
const FFN_GATE_UP_EXPS: &str = "ffn_gate_up_exps";
const FFN_GATE: &str = "ffn_gate";
const FFN_GATE_EXPS: &str = "ffn_gate_exps";
const FFN_UP: &str = "ffn_up";
const FFN_UP_EXPS: &str = "ffn_up_exps";

impl Content<'_> {
    pub(super) fn merge_linear(&mut self, ty: bool) {
        let tensors = std::mem::take(&mut self.tensors);
        if ty {
            let mut collector = MergeCollector::new();
            for (name, tensor) in tensors {
                match collector.collect(&name, tensor) {
                    Collecting::Collected => {}
                    Collecting::Done((name, tensor)) => {
                        self.tensors.insert(name, tensor);
                    }
                    Collecting::Irrelevant(tensor) => {
                        self.tensors.insert(name, tensor);
                    }
                }
            }
            for (name, tensor) in collector.into_iter() {
                self.tensors.insert(name, tensor);
            }
        } else {
            for (name, tensor) in tensors {
                static SPLIT_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(SPLIT).unwrap());
                if let Some(captures) = SPLIT_REGEX.captures(&name) {
                    let pre = &name[..captures.get(0).unwrap().start()];
                    let (_, [name, wb]) = captures.extract();
                    let key = |name: &str| format!("{pre}{name}.{wb}").into();
                    match name {
                        ATTN_QKV => {
                            let [q, k, v] = split_qkv(tensor);
                            self.tensors.insert(key(ATTN_Q), q);
                            self.tensors.insert(key(ATTN_K), k);
                            self.tensors.insert(key(ATTN_V), v);
                        }
                        FFN_GATE_UP => {
                            let [gate, up] = split_gate_up(tensor);
                            self.tensors.insert(key(FFN_GATE), gate);
                            self.tensors.insert(key(FFN_UP), up);
                        }
                        FFN_GATE_UP_EXPS => {
                            let [gate, up] = split_gate_up_exps(tensor);
                            self.tensors.insert(key(FFN_GATE_EXPS), gate);
                            self.tensors.insert(key(FFN_UP_EXPS), up);
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.tensors.insert(name, tensor);
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum Layer {
    Attn,
    Ffn,
    FfnMoe,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum WB {
    Weight,
    Bias,
}

enum Collecting<'a> {
    Collected,
    Done((Cow<'a, str>, Tensor<'a>)),
    Irrelevant(Tensor<'a>),
}

struct MergeCollector<'a>(HashMap<String, GroupCollector<'a>>);
struct GroupCollector<'a>(HashMap<(Layer, WB), [Option<Tensor<'a>>; 3]>);

impl<'a> MergeCollector<'a> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn collect(&mut self, name: &str, tensor: Tensor<'a>) -> Collecting<'a> {
        static MERGE_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(MERGE).unwrap());
        let Some(captures) = MERGE_REGEX.captures(name) else {
            return Collecting::Irrelevant(tensor);
        };
        let pre = &name[..captures.get(0).unwrap().start()];
        let (_, [name, wb]) = captures.extract();
        match self.0.get_mut(pre) {
            Some(group) => group
                .put(name, wb, tensor)
                .map_or(Collecting::Collected, |(name, tensor)| {
                    Collecting::Done((format!("{pre}{name}.{wb}").into(), tensor))
                }),
            None => {
                let mut group = GroupCollector(HashMap::new());
                assert!(group.put(name, wb, tensor).is_none());
                self.0.insert(pre.into(), group);
                Collecting::Collected
            }
        }
    }

    fn into_iter(self) -> impl IntoIterator<Item = (Cow<'a, str>, Tensor<'a>)> {
        self.0.into_iter().flat_map(|(pre, group)| {
            group.0.into_iter().flat_map(move |((layer, wb), tensors)| {
                let wb = match wb {
                    WB::Weight => "weight",
                    WB::Bias => "bias",
                };
                let pre = pre.clone();
                tensors
                    .into_iter()
                    .enumerate()
                    .filter_map(move |(i, tensor)| {
                        tensor.map(|tensor| {
                            let name = match (layer, i) {
                                (Layer::Attn, 0) => ATTN_Q,
                                (Layer::Attn, 1) => ATTN_K,
                                (Layer::Attn, 2) => ATTN_V,
                                (Layer::Ffn, 0) => FFN_GATE,
                                (Layer::Ffn, 1) => FFN_UP,
                                (Layer::FfnMoe, 0) => FFN_GATE_EXPS,
                                (Layer::FfnMoe, 1) => FFN_UP_EXPS,
                                _ => unreachable!(),
                            };
                            (format!("{pre}{name}.{wb}").into(), tensor)
                        })
                    })
            })
        })
    }
}

impl<'a> GroupCollector<'a> {
    fn put(
        &mut self,
        name: &str,
        wb: &str,
        tensor: Tensor<'a>,
    ) -> Option<(&'static str, Tensor<'a>)> {
        let (layer, i) = match name {
            ATTN_Q | "attn.q" => (Layer::Attn, 0),
            ATTN_K | "attn.k" => (Layer::Attn, 1),
            ATTN_V | "attn.v" => (Layer::Attn, 2),
            FFN_GATE => (Layer::Ffn, 0),
            FFN_UP => (Layer::Ffn, 1),
            FFN_GATE_EXPS => (Layer::FfnMoe, 0),
            FFN_UP_EXPS => (Layer::FfnMoe, 1),
            _ => unreachable!(),
        };
        let wb = match wb {
            "weight" => WB::Weight,
            "bias" => WB::Bias,
            _ => unreachable!(),
        };
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        match self.0.entry((layer, wb)) {
            Occupied(mut entry) => {
                entry.get_mut()[i] = Some(tensor);
                match entry.key().0 {
                    Layer::Attn => {
                        if let [Some(_), Some(_), Some(_)] = entry.get() {
                            Some(merge_qkv(entry.remove()))
                        } else {
                            None
                        }
                    }
                    Layer::Ffn => {
                        if let [Some(_), Some(_), None] = entry.get() {
                            Some(merge_gate_up(entry.remove()))
                        } else {
                            None
                        }
                    }
                    Layer::FfnMoe => {
                        if let [Some(_), Some(_), None] = entry.get() {
                            Some(merge_gate_up_exps(entry.remove()))
                        } else {
                            None
                        }
                    }
                }
            }
            Vacant(entry) => {
                let mut array = [None, None, None];
                array[i] = Some(tensor);
                entry.insert(array);
                None
            }
        }
    }
}

fn merge_qkv(tensors: [Option<Tensor>; 3]) -> (&'static str, Tensor) {
    let [Some(q), Some(k), Some(v)] = tensors else {
        unreachable!()
    };
    let [_, qr, _] = distruct(&q);
    let [_, kr, _] = distruct(&k);
    let [_, vr, _] = distruct(&v);
    assert_eq!(qr % kr, 0);
    assert!(qr >= kr);
    assert_eq!(kr, vr);
    (ATTN_QKV, concat(1, [q, k, v]))
}

fn merge_gate_up(tensors: [Option<Tensor>; 3]) -> (&'static str, Tensor) {
    let [Some(gate), Some(up), None] = tensors else {
        unreachable!()
    };
    assert_eq!(gate.shape[1], up.shape[1]);
    (FFN_GATE_UP, concat(1, [gate, up]))
}

fn merge_gate_up_exps(tensors: [Option<Tensor>; 3]) -> (&'static str, Tensor) {
    let [Some(gate), Some(up), None] = tensors else {
        unreachable!()
    };
    (FFN_GATE_UP_EXPS, concat(1, [gate, up]))
}

fn split_qkv(tensor: Tensor) -> [Tensor; 3] {
    let [c, r, _] = distruct(&tensor);
    let rq = c;
    let rkv = (r - c) / 2;
    split(1, tensor, [rq, rkv, rkv])
}

fn split_gate_up(tensor: Tensor) -> [Tensor; 2] {
    let r = tensor.shape[1] / 2;
    split(1, tensor, [r, r])
}

fn split_gate_up_exps(tensor: Tensor) -> [Tensor; 2] {
    let r = tensor.shape[1] / 2;
    split(1, tensor, [r, r])
}

/// 解构形状，补充维度
fn distruct(t: &Tensor) -> [u64; 3] {
    match *t.shape {
        [c] => [c, 1, 1],
        [c, r] => [c, r, 1],
        [c, r, n] => [c, r, n],
        [..] => panic!("invalid tensor shape: {:?}", t.shape),
    }
}

fn concat<const N: usize>(axis: usize, tensors: [Tensor; N]) -> Tensor {
    let ty = tensors[0].ty;
    let mut shape = tensors[0].shape.clone();

    for t in &tensors[1..] {
        assert_eq!(t.ty, ty);
        assert_eq!(t.shape.len(), shape.len());
        for (i, (out, &d)) in zip(&mut shape, &t.shape).enumerate() {
            if i == axis {
                *out += d
            } else {
                assert_eq!(*out, d)
            }
        }
    }

    let shape_ = shape.clone();
    let data = DataPromise::lazy(move || {
        let GGmlTypeSize {
            block_size,
            type_size,
        } = ty.size();
        let group = block_size as usize;
        let unit = type_size as usize;

        let parts = tensors
            .iter()
            .map(|t| {
                let d = t.shape[axis] as usize;
                if axis == 0 {
                    d / group
                } else {
                    d
                }
            })
            .collect::<Vec<_>>();

        let mut ans = MmapMut::map_anon(ty.size().elements_to_bytes(&shape_)).unwrap();
        for (t, out_layout) in zip(tensors, layout(ty, &shape_).split(axis, &parts)) {
            let rearranging = Rearranging::new(&out_layout, &layout(ty, &t.shape), unit).unwrap();
            unsafe { rearranging.launch(ans.as_mut_ptr(), t.data.get().as_ptr()) }
        }
        ans
    });

    Tensor { ty, shape, data }
}

fn split<const N: usize>(axis: usize, tensor: Tensor, split: [u64; N]) -> [Tensor; N] {
    let Tensor { ty, shape, data } = tensor;
    assert_eq!(shape[axis], split.iter().sum());

    let data_layout = layout(ty, &shape);
    let block_size = ty.size().block_size as u64;
    let parts = split.map(|d| if axis == 0 { d / block_size } else { d } as usize);
    let mut data_layout = data_layout.split(axis, &parts);

    split.map(move |d| {
        let mut shape = shape.clone();
        shape[axis] = d;

        let len = ty.size().elements_to_bytes(&shape);
        let unit = ty.size().type_size;

        let dst = layout(ty, &shape);
        let src = data_layout.next().unwrap();
        let rearranging = Rearranging::new(&dst, &src, unit as _).unwrap();

        let data = data.clone();
        let data = DataPromise::lazy(move || {
            let mut ans = MmapMut::map_anon(len).unwrap();
            unsafe { rearranging.launch(ans.as_mut_ptr(), data.get().as_ptr()) }
            ans
        });
        Tensor { ty, shape, data }
    })
}

fn layout(ty: GGmlType, shape: &[u64]) -> ArrayLayout<4> {
    use mem_rearrange::ndarray_layout::Endian::LittleEndian;
    let mut shape = shape.iter().map(|&d| d as _).collect::<Vec<_>>();
    shape[0] /= ty.size().block_size as usize;
    ArrayLayout::new_contiguous(&shape, LittleEndian, ty.size().type_size as _)
}
