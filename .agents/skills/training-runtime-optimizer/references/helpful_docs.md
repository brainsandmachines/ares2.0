# Helpful External Docs

These documents are useful for candidate ideas. They do not override the repo contract.

- PyTorch Performance Tuning Guide  
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

- `torch.compile` tutorial  
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

- AMP examples  
  https://docs.pytorch.org/docs/2.9/notes/amp_examples.html

- Pin memory and non-blocking transfers  
  https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

- Channels-last memory format  
  https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html

- Data loading tutorial  
  https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

Useful ideas from those docs for this repo:

- tune DataLoader worker behavior
- prefer pinned memory and non-blocking transfers when already compatible
- test `channels_last` on ConvNeXt workloads
- test `torch.compile` only as an additive candidate
- test `zero_grad(set_to_none=True)` through additive wrapping
- always benchmark on the real protocol path before promoting a candidate
