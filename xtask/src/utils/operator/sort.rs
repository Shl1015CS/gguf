use super::Content;
use itertools::Itertools;
use regex::Regex;
use std::{cmp::Ordering, collections::HashMap, sync::LazyLock};

impl Content<'_> {
    pub(super) fn sort_tensors(&mut self) {
        let tensors = std::mem::take(&mut self.tensors);
        self.tensors = tensors
            .into_iter()
            .map(|(k, v)| (Name::new_key(&k), k, v))
            .sorted_unstable_by(|(a, ..), (b, ..)| a.cmp(b))
            .map(|(_, k, v)| (k, v))
            .collect();
    }
}

const MID: &[&str] = &[
    "attn_norm",
    "attn_norm_2",
    "ln1",
    "attn_qkv",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn.q",
    "attn.k",
    "attn.v",
    "attn_output",
    "attn_out",
    "attn.out",
    "ffn_norm",
    "ln2",
    "ffn_gate_inp",
    "ffn_gate_up",
    "ffn_gate_up_exps",
    "ffn_gate_up_exp",
    "ffn_gate",
    "ffn_gate_exps",
    "ffn_gate_exp",
    "ffn_up",
    "ffn_up_exps",
    "ffn_up_exp",
    "ffn_down",
    "ffn_down_exps",
    "ffn_down_exp",
];

const POST: &[&str] = &["weight", "bias"];

#[test]
fn test() {}

#[derive(PartialEq, Eq, Debug)]
struct Name<'a>(Pre<'a>, Mid<'a>, Post<'a>);

#[derive(PartialEq, Eq, Debug)]
struct Pre<'a>(Vec<PreSeg<'a>>);
#[derive(PartialEq, Eq, Debug)]
struct Mid<'a>(&'a str);
#[derive(PartialEq, Eq, Debug)]
struct Post<'a>(&'a str);

#[derive(PartialEq, Eq, Debug)]
enum PreSeg<'a> {
    Str(&'a str),
    Num(usize),
}

impl Name<'static> {
    fn new_key(value: &str) -> Self {
        static REGEX: LazyLock<Regex> = LazyLock::new(|| {
            let mut mid = String::new();
            for name in MID {
                for c in name.chars() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        mid.push(c)
                    } else if c == '.' {
                        mid.push_str(r"\.")
                    } else {
                        panic!("invalid char: {c}")
                    }
                }
                mid.push('|')
            }
            mid.pop();
            Regex::new(&mid).unwrap()
        });

        let value = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(value.as_ptr(), value.len()))
        };
        let (start, end) = REGEX
            .find(value)
            .map_or((value.len(), value.len()), |mid| (mid.start(), mid.end()));
        let pre = value[..start]
            .split('.')
            .map(|s| s.parse::<usize>().map_or(PreSeg::Str(s), PreSeg::Num))
            .collect();
        let mid = &value[start..end];
        let post = &value[end..];
        Self(Pre(pre), Mid(mid), Post(post))
    }
}

impl Ord for Name<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::Equal;
        match self.0.cmp(&other.0) {
            Equal => match self.1.cmp(&other.1) {
                Equal => self.2.cmp(&other.2),
                ord => ord,
            },
            ord => ord,
        }
    }
}
impl Ord for Pre<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::{Equal, Greater, Less};

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            match (a, b) {
                (PreSeg::Str(_), PreSeg::Num(_)) => return Less,
                (PreSeg::Num(_), PreSeg::Str(_)) => return Greater,
                (PreSeg::Str(a), PreSeg::Str(b)) => match a.cmp(b) {
                    Equal => {}
                    ord => return ord,
                },
                (PreSeg::Num(a), PreSeg::Num(b)) => match a.cmp(b) {
                    Equal => {}
                    ord => return ord,
                },
            }
        }
        self.0.len().cmp(&other.0.len())
    }
}

impl Ord for Mid<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        static ORDER_MAP: LazyLock<HashMap<&str, usize>> =
            LazyLock::new(|| MID.iter().enumerate().map(|(i, s)| (*s, i)).collect());
        cmp_by_map(self.0, other.0, &ORDER_MAP)
    }
}

impl Ord for Post<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::{Equal, Greater, Less};

        static ORDER_MAP: LazyLock<HashMap<&str, usize>> =
            LazyLock::new(|| POST.iter().enumerate().map(|(i, s)| (*s, i)).collect());
        let mut a = self.0.split('.');
        let mut b = other.0.split('.');
        loop {
            match (a.next(), b.next()) {
                (Some(a), Some(b)) => match cmp_by_map(a, b, &ORDER_MAP) {
                    Equal => {}
                    ord => break ord,
                },
                (Some(_), None) => break Greater,
                (None, Some(_)) => break Less,
                (None, None) => break Equal,
            }
        }
    }
}

fn cmp_by_map(a: &str, b: &str, map: &HashMap<&str, usize>) -> Ordering {
    match (map.get(a), map.get(b)) {
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (Some(a), Some(b)) => a.cmp(b),
        (None, None) => a.cmp(b),
    }
}

impl PartialOrd for Name<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for Pre<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for Mid<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for Post<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
