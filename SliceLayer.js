/*
入力テンソル [batch,seqLen,embeddedDim]から指定されたseqLenのみを取り出すレイヤ
*/
class SliceLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.startIndex = config.startIndex ?? 0;
        this.size = config.size ?? 1;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], this.size, inputShape[2]];
    }

    call(inputs) {
        const x = Array.isArray(inputs) ? inputs[0] : inputs;
        return x.slice([0, this.startIndex, 0], [-1, this.size, -1]);
    }
    static get className() {
        return 'SliceLayer';
    }
}

// Register the class for serialization
tf.serialization.registerClass(SliceLayer);
