# QuickBitsNN
Neural Networks trained on Blockchain Data to generate values needed for new block creation.


The minting of new blocks for the blockchain requires hashes of the block headers to have a certain number of leading zeros.
A "QuickBit" is the set of six parameters which are concatenated into the Block Header before being fed into the hash function.
The last of these six parameters is the "nonce" and traditionally mined at random over its range (uint32) in order to find an acceptable hash.
This code attempts to predict "nonce" values using generative AI models: Transformer and VAE Convolutional Network.


The method can be into these main parts:
1. Sniff out the QuickBit blockchain data on the Blockchain Core.
2. Train the models using QuickBit data.
3. Generate "nonce" and evaluate the acceptance of Block Header.


In theory, the neural networks can learn the underlying block acceptance criteria and generate blocks faster than mining.
Transformers can use Encoder-Decoder architecture to directly translate from Header to Hash.
Data represented as QuickBits provides the model with information directly corrolated to the Hash without actually performing the Hashing function.
This is extemely important since the Hash function is not differentiable and cannot be backpropogated through.
The Hash can be completely avoided by using a decoder-only stack, (similar to GPT), to generate the blocktail through autoregression.
In the case of the Variational Auto Encoder, the Blocktail is generated along with the reconstruction of the full Block Header.



The models use a tokenization schema of the "Hashex" language dictionary.
Hashex is composed of the hex alphabet (0-9,a-f) and each of the 256 words corresponds to the total permutations for 1 byte (2 hex alphabet letters).
The block headers are 160 ascii characters long but are only 80 Hashex tokens.



