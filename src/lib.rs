#![feature(optin_builtin_traits)]

extern crate cudart;
//extern crate memrepr;

use crate::ctx::{GpuCtxGuard};

use cudart::{CudaStream, cuda_alloc_device, cuda_free_device};

use std::mem::{size_of};
use std::ops::{Deref};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod ctx;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GpuDev(pub i32);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum GpuDom {
  Host,
  Dev(GpuDev),
}

pub trait GpuDelay {
  type Data: Copy + 'static;
}

// FIXME: implement `GpuDelay` for `Copy` types?

impl GpuDelay for () {
  type Data = ();
}

pub trait GpuDelayed<V: GpuDelay>: Deref<Target=V> {
  fn domain(&self) -> GpuDom;
  fn delayed_ptr(&self) -> *const <V as GpuDelay>::Data;
}

pub trait GpuDelayedMut<V: GpuDelay>: GpuDelayed<V> {
  fn delayed_ptr_mut(&self) -> *mut <V as GpuDelay>::Data;
}

pub trait GpuRegion<T: Copy + 'static> {
  fn device(&self) -> GpuDev;
  fn as_devptr(&self) -> *const T;
  fn region_len(&self) -> usize;
}

pub trait GpuRegionMut<T: Copy + 'static>: GpuRegion<T> {
  fn as_devptr_mut(&self) -> *mut T;
}

pub struct GpuPinnedMem<T: Copy + 'static> {
  ptr:  *mut T,
  len:  usize,
}

impl<T: Copy + 'static> GpuDelay for GpuPinnedMem<T> {
  type Data = T;
}

impl<T: Copy + 'static> Drop for GpuPinnedMem<T> {
  fn drop(&mut self) {
    // TODO
    unimplemented!();
  }
}

impl<T: Copy + 'static> GpuPinnedMem<T> {
  pub unsafe fn alloc(len: usize) -> GpuPinnedMem<T> {
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

pub struct GpuVMem<T: Copy + 'static> {
  dev:  GpuDev,
  dptr: *mut T,
  len:  usize,
}

impl<T: Copy + 'static> GpuDelay for GpuVMem<T> {
  type Data = T;
}

impl<T: Copy + 'static> GpuRegion<T> for GpuVMem<T> {
  fn device(&self) -> GpuDev {
    self.dev
  }

  fn as_devptr(&self) -> *const T {
    self.dptr
  }

  fn region_len(&self) -> usize {
    self.len
  }
}

impl<T: Copy + 'static> GpuRegionMut<T> for GpuVMem<T> {
  fn as_devptr_mut(&self) -> *mut T {
    self.dptr
  }
}

impl<T: Copy + 'static> Drop for GpuVMem<T> {
  fn drop(&mut self) {
    let _ctx = GpuCtxGuard::new(self.dev);
    // NB: This _will_ implicitly synchronize; see:
    // <https://github.com/thrust/thrust/issues/905>
    // <https://cs.unc.edu/~tamert/papers/ecrts18b.pdf>
    unsafe {
      match cuda_free_device(self.dptr) {
        Err(e) => panic!("cudaFree failed: {:?} ({})", e, e.get_string()),
        Ok(_) => {}
      }
    }
  }
}

impl<T: Copy + 'static> GpuVMem<T> {
  pub unsafe fn alloc(len: usize, dev: GpuDev) -> GpuVMem<T> {
    let _ctx = GpuCtxGuard::new(dev);
    // NB: This _may_ implicitly synchronize; see:
    // <https://github.com/thrust/thrust/issues/905>
    let dptr = match cuda_alloc_device(len) {
      Err(e) => panic!("cudaMalloc failed: {:?} ({})", e, e.get_string()),
      Ok(dptr) => dptr,
    };
    GpuVMem{
      dev,
      dptr,
      len,
    }
  }
}

/*impl<T: ZeroBits + Copy + 'static> GpuVMem<T> {
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
