# QuickBitsNN
Neural Networks trained on Blockchain Data to generate Nonce values.


The minting of new blocks for the blockchain requires hashes of the block headers to have a certain number of leading zeros.
A "QuickBit" is the set of six parameters which are concatenated into a Preblock Header before being fed into the hash function.
Of these six parameters, the Nonce is traditionally mined at random over its range (uint32) in order to find an acceptable hash.
This code attempts to predict Nonce values using generative AI models: Transformer and Deep Convolutional Network.


The method can be into these main parts:
1. Sniff out the QuickBit blockchain data on the BitCoin Core
2. Train the models using QuickBit data
3. Generate Nonces and Evaluate leading zeros of hash


The models use a tokenization schema of the "Hashex" language dictionary.
Hashex is composed of the hex alphabet (0-9,a-f) and each of the 256 words corresponds to the total permutations for 1 byte (2 hex alphabet letters).
The QuickBits are converted into their Preblock Headers, then Tokenized.
This provides the model with data which is closest to the hash without actually performing the hashing function.
In theory, the neural networks will learn the underlying patterns which can produce high numbers of leading hash zeros and generate Nonces for new blocks faster than mining. 


Two models requires two different approaches to training.
- The transformer uses Cross Entropy Loss against a set of logits (4 digits with 256 possible letters)
- The Convolutional Net abstracts the tokenization representation into an R,G,B,A image and uses reinforced learning to predict the next nonce state.





