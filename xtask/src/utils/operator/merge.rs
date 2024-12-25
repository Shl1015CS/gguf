use super::{super::Tensor, Content, DataPromise};
use ggus::DataFuture;
use memmap2::MmapMut;
use regex::Regex;
use std::{borrow::Cow, collections::HashMap, hash::Hash, sync::LazyLock};

const MERGE: &str =
    r"(attn\.q|attn_q|attn\.k|attn_k|attn\.v|attn_v|ffn_gate|ffn_up)\.(weight|bias)$";
const SPLIT: &str = r"(attn_qkv|ffn_gate_up)\.(weight|bias)$";
const ATTN_QKV: &str = "attn_qkv";
const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const FFN_GATE_UP: &str = "ffn_gate_up";
const FFN_GATE: &str = "ffn_gate";
const FFN_UP: &str = "ffn_up";

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
                    match name {
                        ATTN_QKV => {
                            let [q, k, v] = split_qkv(tensor);
                            self.tensors.insert(format!("{pre}{ATTN_Q}.{wb}").into(), q);
                            self.tensors.insert(format!("{pre}{ATTN_K}.{wb}").into(), k);
                            self.tensors.insert(format!("{pre}{ATTN_V}.{wb}").into(), v);
                        }
                        FFN_GATE_UP => {
                            let [gate, up] = split_gate_up(tensor);
                            self.tensors
                                .insert(format!("{pre}{FFN_GATE}.{wb}").into(), gate);
                            self.tensors
                                .insert(format!("{pre}{FFN_UP}.{wb}").into(), up);
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
    let qr = q.shape.get(1).copied().unwrap_or(1);
    let kr = k.shape.get(1).copied().unwrap_or(1);
    let vr = v.shape.get(1).copied().unwrap_or(1);
    assert_eq!(qr % kr, 0);
    assert!(qr >= kr);
    assert_eq!(kr, vr);
    (ATTN_QKV, concat1([q, k, v]))
}

fn merge_gate_up(tensors: [Option<Tensor>; 3]) -> (&'static str, Tensor) {
    let [Some(gate), Some(up), None] = tensors else {
        unreachable!()
    };
    assert_eq!(gate.shape[1], up.shape[1]);
    (FFN_GATE_UP, concat1([gate, up]))
}

fn split_qkv(tensor: Tensor) -> [Tensor; 3] {
    let [c, r, _] = distruct(&tensor);
    let rq = c;
    let rkv = (r - c) / 2;
    split1(tensor, [rq, rkv, rkv])
}

fn split_gate_up(tensor: Tensor) -> [Tensor; 2] {
    let r = tensor.shape[1] / 2;
    split1(tensor, [r, r])
}

/// 解构形状，补充分布维度
fn distruct(t: &Tensor) -> [u64; 3] {
    match *t.shape {
        [c] => [c, 1, 1],
        [c, r] => [c, r, 1],
        [c, r, n] => [c, r, n],
        [..] => panic!("invalid tensor shape: {:?}", t.shape),
    }
}

/// 构造形状，去除分布维度
fn construct(c: u64, r: u64, n: u64) -> Vec<u64> {
    if n == 1 {
        vec![c, r]
    } else {
        vec![c, r, n]
    }
}

/// 在最高维分割数据
macro_rules! split0 {
    ($s:expr; $d:expr; [$i: expr]) => {
        $s[$d * $i..][..$d]
    };
}

fn concat1<const N: usize>(tensors: [Tensor; N]) -> Tensor {
    // 提取数据类型和形状
    let ty = tensors[0].ty;
    let [c, mut r, n] = distruct(&tensors[0]);
    for t in &tensors[1..] {
        let [c_, r_, n_] = distruct(t);
        assert_eq!(c, c_);
        assert_eq!(n, n_);
        r += r_;
    }
    // 锁定形状和数据
    let r = r;
    let data = tensors.map(|t| t.data);
    // 生成张量
    Tensor {
        ty,
        shape: construct(c, r, n),
        data: DataPromise::lazy(move || {
            let data: [_; N] = std::array::from_fn(|i| data[i].get());

            let len = data.iter().map(|s| s.len()).sum();
            assert_eq!(len, ty.size().elements_to_bytes(&[c, r, n]));

            let n = n as _;
            let mut ans = MmapMut::map_anon(len).unwrap();
            for i in 0..n {
                let mut dst = &mut split0!(ans; len / n; [i]);
                for data in data {
                    let data = &split0!(data; data.len() / n; [i]);
                    let (dst_, tail) = dst.split_at_mut(data.len());
                    dst_.copy_from_slice(data);
                    dst = tail;
                }
                assert!(dst.is_empty());
            }
            ans
        }),
    }
}

fn split1<const N: usize>(tensor: Tensor, split: [u64; N]) -> [Tensor; N] {
    // 提取数据类型和形状
    let ty = tensor.ty;
    let [c, r, n] = distruct(&tensor);
    assert_eq!(r, split.iter().sum());
    // 计算规模
    let size = ty.size();
    let d = size.elements_to_bytes(&[c, r]);
    // 生成张量
    let mut presum = 0;
    split.map(|r_| {
        let d_ = size.elements_to_bytes(&[c, r_]);
        let data = tensor.data.clone();
        let presum_ = presum;
        presum += d_;
        Tensor {
            ty,
            shape: construct(c, r_, n),
            data: DataPromise::lazy(move || {
                let n = n as _;
                let data = data.get();
                assert_eq!(data.len(), d * n);

                let mut ans = MmapMut::map_anon(d_ * n).unwrap();
                for i in 0..n {
                    let src = &split0!(data; d; [i]);
                    let dst = &mut split0!(ans; d_; [i]);
                    dst.copy_from_slice(&src[presum_..][..d_]);
                }
                ans
            }),
        }
    })
}
