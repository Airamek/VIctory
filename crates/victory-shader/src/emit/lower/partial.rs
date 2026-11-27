//! A partial dispatch's thread-count guard (upstream #3426, dispatch
//! half only): the ray-structure builder's rehydration kernels encode one
//! header per active thread of the group, so their last group can be short
//! and the caller's dispatch knows the actual thread count.

use super::{Lower, Val};
use crate::emit::UserData;
use crate::emit::bitcode::{Bin, Term, V, pred};
use crate::emit::shader::USER_DISPATCH_THREADS_X;
use crate::native::RootType;

impl Lower<'_> {
    /// A user-data dword the host fills as `root`/`slot`/`dword` say, made
    /// once per distinct triple: the first free index past the layout's
    /// user SGPRs. Returns its load in the entry block.
    pub(super) fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        let known = self
            .user_reads
            .values()
            .find(|u| u.root == root && u.slot == slot && u.dword == dword)
            .copied();
        let u = match known {
            Some(u) => u,
            None => {
                let mut index = self
                    .x
                    .native
                    .layout
                    .iter()
                    .map(|e| u32::from(e.user_data) + e.sgprs)
                    .max()
                    .unwrap_or(0);
                while self.user_reads.contains_key(&index) {
                    index += 1;
                }
                let u = UserData {
                    index,
                    slot,
                    root,
                    dword,
                };
                self.user_reads.insert(index, u);
                u
            }
        };
        self.user_at(u)
    }

    /// Guards the entry block when `partial`: threads whose global X index
    /// is at or past the host's `USER_DISPATCH_THREADS_X` count do nothing
    /// (a zero count keeps every thread active). The guard reads the group
    /// id and the lane id in the entry block, before any wave or LDS use.
    pub(super) fn partial_dispatch_prologue(&mut self, partial: bool) {
        if !partial {
            return;
        }
        let original = self.s.entry;
        let width = self.s.threads.expect("compute thread count")[0];
        let limit = self.host_word(RootType::Other(USER_DISPATCH_THREADS_X), 0, width);
        let Val::I32(group) = self.compute_id("groupId.i32", 94, 0) else {
            unreachable!()
        };
        let Val::I32(thread) = self.thread_id(0) else {
            unreachable!()
        };
        let width = self.s.m.i32(width);
        let first = self.s.m.bin(original, Bin::Mul, group, width);
        let x = self.s.m.bin(original, Bin::Add, first, thread);
        let zero = self.s.m.i32(0);
        let unlimited = self.s.m.cmp(original, pred::EQ, limit, zero);
        let inside = self.s.m.cmp(original, pred::IULT, x, limit);
        let active = self.s.m.bin(original, Bin::Or, unlimited, inside);
        let body = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(original, Term::CondBr(active, body, done));
        self.s.m.terminate(done, Term::Ret);
        self.s.entry = body;
    }
}
