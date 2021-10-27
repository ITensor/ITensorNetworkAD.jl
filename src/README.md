Definition:

* TreeTensor: A tensor implicitly represented by a tree structure.

* MPSTensor: A tensor implicitly represented by a MPS structure.

* SubNetwork: A group of tensors. Each tensor has type ITensor, or AbstractTensor (can be TreeTensor or MPSTensor).

We need to think of TreeTensor and MPSTensor as objects similar to ITensor, and representing an individual tensor. SubNetwork is a group of tensors.
