// const tf = require('@tensorflow/tfjs');

/**
 * MultiHeadAttention Layer
 * 
 * References:
 * https://arxiv.org/abs/1706.03762
 */
class MultiHeadAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.numHeads = config.numHeads;
        this.keyDim = config.keyDim;
        this.valueDim = config.valueDim ?? this.keyDim;
        this.outputDim = config.outputDim; // Optional
        this.useBias = config.useBias === undefined ? true : config.useBias;

        this.keepAttentionScores = config.keepAttentionScores ?? false;
    }

    computeOutputShape(inputShape) {
        // console.log('computeOutputShape called with:', JSON.stringify(inputShape));
        // inputShape is array of shapes: [query, value, key] or single shape
        const queryShape = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
        const batchSize = queryShape[0];
        const seqLen = queryShape[1];

        // If outputDim is not specified, projecting back to query's last dimension
        const lastDim = this.outputDim ? this.outputDim : queryShape[queryShape.length - 1];

        return [batchSize, seqLen, lastDim];
    }

    build(inputShape) {
        // console.log('build called with:', JSON.stringify(inputShape));
        const queryShape = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
        const inputDim = queryShape[queryShape.length - 1];

        // If key/value shapes are different (e.g. cross attention), we should handle that,
        // but for simplicity assuming the last dimension of key/value maps to inputDim or we just define weights based on inputDim.
        // In Keras implementation, they look at keyShape/valueShape.
        let keyShape = queryShape;
        let valueShape = queryShape;
        if (Array.isArray(inputShape[0])) {
            // inputs = [query, value, key]
            if (inputShape.length > 1) valueShape = inputShape[1];
            if (inputShape.length > 2) keyShape = inputShape[2];
        }

        const keyInputDim = keyShape[keyShape.length - 1];
        const valueInputDim = valueShape[valueShape.length - 1];

        if (!this.outputDim) {
            this.outputDim = inputDim;
        }

        // Query Projection
        this.kernelQ = this.addWeight(
            'kernel_q',
            [inputDim, this.numHeads * this.keyDim],
            'float32',
            tf.initializers.glorotUniform(),
            null,
            true
        );
        if (this.useBias) {
            this.biasQ = this.addWeight(
                'bias_q',
                [this.numHeads * this.keyDim],
                'float32',
                tf.initializers.zeros(),
                null,
                true
            );
        }

        // Key Projection
        this.kernelK = this.addWeight(
            'kernel_k',
            [keyInputDim, this.numHeads * this.keyDim],
            'float32',
            tf.initializers.glorotUniform(),
            null,
            true
        );
        if (this.useBias) {
            this.biasK = this.addWeight(
                'bias_k',
                [this.numHeads * this.keyDim],
                'float32',
                tf.initializers.zeros(),
                null,
                true
            );
        }

        // Value Projection
        this.kernelV = this.addWeight(
            'kernel_v',
            [valueInputDim, this.numHeads * this.valueDim],
            'float32',
            tf.initializers.glorotUniform(),
            null,
            true
        );
        if (this.useBias) {
            this.biasV = this.addWeight(
                'bias_v',
                [this.numHeads * this.valueDim],
                'float32',
                tf.initializers.zeros(),
                null,
                true
            );
        }

        // Output Projection
        this.kernelO = this.addWeight(
            'kernel_o',
            [this.numHeads * this.valueDim, this.outputDim],
            'float32',
            tf.initializers.glorotUniform(),
            null,
            true
        );
        if (this.useBias) {
            this.biasO = this.addWeight(
                'bias_o',
                [this.outputDim],
                'float32',
                tf.initializers.zeros(),
                null,
                true
            );
        }

        this.built = true;
    }

    call(inputs, kwargs) {
        // const [query, key, value] = inputs;
        const query = inputs[0].clone();
        const key = inputs[1].clone();
        const value = inputs[2].clone();

        const batchSize = query.shape[0] || -1;
        // Note: In tfjs-layers, batchSize might be null during build, but we are in call() now.

        // Helper for 3D matmul (BS, D) x (D, O) -> (BS, O)
        // Reshapes to (B*S, D) -> matmul -> (B*S, O) -> (B, S, O)
        const safeMatMul = (t, w) => {
            const seq = t.shape[1];
            const d = t.shape[2];
            // if t rank is 2 (e.g. [batch, dim]), just matmul
            if (t.shape.length === 2) {
                return tf.matMul(t, w);
            }
            const flat = tf.reshape(t, [-1, d]);
            const res = tf.matMul(flat, w);
            return tf.reshape(res, [-1, seq, w.shape[1]]);
        };

        // 1. Project Q, K, V
        let Q = safeMatMul(query, this.kernelQ.read());
        if (this.biasQ) Q = tf.add(Q, this.biasQ.read());

        let K = safeMatMul(key, this.kernelK.read());
        if (this.biasK) K = tf.add(K, this.biasK.read());

        let V = safeMatMul(value, this.kernelV.read());
        if (this.biasV) V = tf.add(V, this.biasV.read());

        // 2. Split Heads
        // Current shape: [batch, seq_len, numHeads * dim]
        // Target shape: [batch, num_heads, seq_len, dim]

        Q = this.splitHeads(Q, batchSize, this.numHeads, this.keyDim);
        K = this.splitHeads(K, batchSize, this.numHeads, this.keyDim);
        V = this.splitHeads(V, batchSize, this.numHeads, this.valueDim);

        // 3. Scaled Dot-Product Attention
        // Q: [batch, heads, seqQ, dk]
        // K: [batch, heads, seqK, dk]
        // K_T: [batch, heads, dk, seqK]
        // scores = Q * K^T

        // tf.matMul supports broadcasting for batch and heads if rank >= 3
        // matMul(a, b, transposeA, transposeB)
        let scores = tf.matMul(Q, K, false, true);
        // scores shape: [batch, heads, seqQ, seqK]

        // Scale
        const scale = tf.scalar(Math.sqrt(this.keyDim));
        scores = tf.div(scores, scale);

        // Optional: Masking (not implemented for simplicity, but place for it is here)
        // if (mask) { ... }

        // Softmax shape [batch, heads, inputDim, inputDim]
        const attentionWeights = tf.softmax(scores, -1); // last dim
        if (this.keepAttentionScores) {
            // console.log("attn", attentionWeights.shape)
            this.attentionScores = attentionWeights.arraySync();
        }
        // console.log(attentionWeights)

        // 4. Context
        // weights: [batch, heads, seqQ, seqK]
        // V: [batch, heads, seqK, dv]
        // context = weights * V
        let context = tf.matMul(attentionWeights, V);
        // context shape: [batch, heads, seqQ, dv]

        // 5. Combine Heads
        // Target: [batch, seqQ, heads * dv]
        context = this.combineHeads(context);

        // 6. Output projection
        let output = safeMatMul(context, this.kernelO.read());
        if (this.biasO) output = tf.add(output, this.biasO.read());

        // tf.layers.addの問題はカスタムレイヤーでtf.tidyを使っていたのが原因だったもよう。
        // そしてそのaddを外し、sliceも外して、通常のmultiheadattentionの出力に戻した。
        // よってこの出力に残差を付ける場合、このレイヤのあとで、tf.layers.add().apply([output,query])をする。
        return output;
    }

    splitHeads(x, batchSize, numHeads, dim) {
        // x: [batch, seq_len, numHeads * dim]
        // reshape to [batch, seq_len, numHeads, dim]
        // transpose to [batch, numHeads, seq_len, dim]

        // Note: x.shape might use -1 for batch if unknown, but we need concrete values for reshape unless we use -1.
        // In call(), batchSize should be concrete number from x.shape[0].
        // However, x.shape[0] might be null in symbolic execution (not typical in imperative tfjs).
        // We'll trust x.shape

        const seqLen = x.shape[1];
        const reshaped = tf.reshape(x, [-1, seqLen, numHeads, dim]);
        return tf.transpose(reshaped, [0, 2, 1, 3]);
    }

    combineHeads(x) {
        // x: [batch, numHeads, seq_len, dim]
        // transpose to [batch, seq_len, numHeads, dim]
        // reshape to [batch, seq_len, numHeads * dim]
        const batchSize = x.shape[0] || -1;
        const numHeads = x.shape[1];
        const seqLen = x.shape[2];
        const dim = x.shape[3];

        const transposed = tf.transpose(x, [0, 2, 1, 3]);
        return tf.reshape(transposed, [-1, seqLen, numHeads * dim]);
    }

    getKeepAttentionScores() {
        return this.keepAttentionScores;
    }
    setKeepAttentionScores(value) {
        this.keepAttentionScores = value;
    }
    getAttentionScores() {
        return tf.tidy(() => {
            // [batch, numHeads, dim, dim]
            const t = tf.tensor(this.attentionScores);
            // batchごとにすべてのheadを一塊にして0-1に標準化
            const mi = t.min([1, 2, 3], true);
            const mx = t.max([1, 2, 3], true);
            return t.sub(mi).div(mx.sub(mi).add(1e-8)).arraySync();
            // console.log("att",this.attentionScores)
            // const summed = tf.sum(this.attentionScores, 0);
            // console.log(summed.shape)
            // const [mx, mi] = [summed.max(), summed.min()];
            // return summed.sub(mi).div(mx.sub(mi)).arraySync();
        });
    }

    static get className() {
        return 'MultiHeadAttention';
    }
}

// Register the class for serialization
tf.serialization.registerClass(MultiHeadAttention);

// module.exports = MultiHeadAttention;

// window.MultiHeadAttention = MultiHeadAttention;
// console.log(MultiHeadAttention)