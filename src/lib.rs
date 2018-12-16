#![feature(optin_builtin_traits)]

extern crate cudart;
//extern crate memrepr;

use cudart::{CudaStream, cuda_memset_async};
//use memrepr::{ZeroBits};

use std::mem::{size_of};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod ctx;

pub trait GpuDelay {
  type Target: Copy + 'static;
}

pub trait GpuDelayed<V: GpuDelay> {
  unsafe fn dptr(&self) -> *const V::Target;
}

pub trait GpuDelayedMut<V: GpuDelay>: GpuDelayed<V> {
  unsafe fn dptr_mut(&self) -> *mut V::Target;
}

pub struct GpuUnsafePinnedMem<T: Copy + 'static> {
  ptr:  *mut T,
  len:  usize,
  bysz: usize,
}

impl<T: Copy + 'static> GpuDelay for GpuUnsafePinnedMem<T> {
  type Target = T;
}

impl<T: Copy + 'static> GpuUnsafePinnedMem<T> {
  pub unsafe fn alloc(len: usize) -> GpuUnsafePinnedMem<T> {
    // TODO
    unimplemented!();
  }

  pub unsafe fn as_slice(&self) -> &[T] {
    from_raw_parts(self.ptr, self.len)
  }

  pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
    from_raw_parts_mut(self.ptr, self.len)
  }
}

pub struct GpuUnsafeMem<T: Copy + 'static> {
  dptr: *mut T,
  len:  usize,
  bysz: usize,
  dev:  i32,
}

impl<T: Copy + 'static> GpuDelay for GpuUnsafeMem<T> {
  type Target = T;
}

/*impl<T: ZeroBits + Copy + 'static> GpuUnsafeMem<T> {
  pub unsafe fn set_zeros(&self, stream: &mut CudaStream) {
    let res = cuda_memset_async(
        self.dptr as *mut u8,
        0,
        self.len * size_of::<T>(),
        stream,
    );
    match res {
      Err(_) => panic!(),
      Ok(_) => {}
    }
  }
}*/

impl<T: Copy + 'static> Drop for GpuUnsafeMem<T> {
  fn drop(&mut self) {
    // TODO
    unimplemented!();
  }
}

impl<T: Copy + 'static> GpuUnsafeMem<T> {
  pub unsafe fn alloc(len: usize, dev: i32) -> GpuUnsafeMem<T> {
    // TODO
    unimplemented!();
  }

  pub unsafe fn as_devptr(&self) -> *const T {
    self.dptr
  }

  pub unsafe fn as_devptr_mut(&self) -> *mut T {
    self.dptr
  }
}
