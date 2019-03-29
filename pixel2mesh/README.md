# INSTRUCTIONS
## CONSTRUCTING THE **FEED DICT**
### Construct feed dictionary by pickle load importing the .dat composed by two parts
0. pkl\[0]: coordinates (name='features')
1. pkl\[1]: support 1
2. pkl\[2]: support 2
3. pkl\[3]: support 3
4. pkl\[4]: pool_idx
5. pkl\[5]: faces
6. pkl\[6]: unknown
7. pkl\[7]: laplacian_normalization

### pkl
0. \[0]: coordinates:
    - shape: (156, 3), 156 vertices with 3-dim coordinates

1. \[1]: edge, feed into block 1, define the convolution for the update the values 

    **Note**: this is a sparse tensor represented in three dense tensors
    - \[0]:
        ```
        [0]: shape (156, 2) int, all composed by [n, n], n from 0 to 155
        [1]: shape (156,), = ones(156, 1) float
        [2]: = [156, 156]
        ```
    - \[1]:
        ```
        [0]: shape (1080, 2) int
        [1]: shape (1080,) float
        [2]: = [156, 156]
        ```

2. \[2]: edge, feed into block 2, define the convolution for the update the values 
    - \[0]:
        ```
        [0]: shape (618, 2) int, all composed by [n, n], n from 0 to 617
        [1]: shape (156,), = ones(618, 1) float
        [2]: [618, 618]
        ```
    - \[1]:
        ``` 
        [0]: shape (4314, 2) int
        [1]: shape (4314,) float
        [2]: = [618, 618]
        ```

3. \[3]: edge, feed into block 3, define the convolution for the update the values 
    - \[0]:
        ```
        [0]: shape(2466, 2) int, all composed by [n, n], n from 1 to 156
        [1]: shape(2466,) = ones(2466, 1)
        [2]: = [2466, 2466]
        ```
    - \[1]:
        ```
        [0]: shape (17250, 2) int
        [1]: shape (17250,) float
        [2]: = [2466, 2466]
        ```

4. \[4]: pool_idx
    ```
    [0]: shape (462, 2) int, seems like coordinates on 2D
    [1]: shape (1848, 2) int, seems like coordinates on 2D
    ```

5. \[5]: faces
    ```
    [0]: shape (462, 4) int
    [1]: shape (1848, 4) int
    [2]: shape (7392, 4) int
    ```

6. \[6]: unknown
    ```
    [0]: shape (156, 3) float. very small number (first block)
    [1]: shape (618, 3) float, very small number (NOT 628) (second block)
    [2]: shape (2466, 4) flaot, very small number (third block)
    ```

7. \[7]: laplacian normalization
    ```
    [0]: shape (156, 10) int, [-3] and [-4] are always -1 
    [1]: shape (618, 10) int
    [2]: shape (2466, 10) int
    ```

## BUILD THE **MODEL**
### Basic *Model*  Module:
1. First to go through function \_\_init_\_, get a series of blank properties including name, logging, vars, placeholders, payers, activations, inputs, outputs, loss, optimizer, opt_op.

2. 

## MULTITHREAD DATA PROCESSING **FETCHER**
### Inherit from threading.Thread
Use 
