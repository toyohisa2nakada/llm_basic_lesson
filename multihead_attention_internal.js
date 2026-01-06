let datasets = undefined;
let models = undefined;

function setupVisor({ tfvis, datasetsMode = "default", setDatasets, learn }) {
    tfvis.visor();
    const visorElement = document.querySelector(".visor");
    visorElement.style.width = "100%";
    visorElement.style.left = "0";

    [".visor-controls", ".visor-tabs"].forEach((e) => {
        const elem = visorElement.querySelector(e);
        elem.style.display = "none";
    });

    const learningPanelElem = tfvis.visor().surface({ name: "コントロールパネル" });

    const panel = document.createElement("div");
    learningPanelElem.container.appendChild(panel);

    const selElem = document.createElement("span");
    selElem.innerHTML = `<select id="trainingData" name="trainingData"><option value="favorite">好き嫌いデータ</option><option value="homonym" selected>同音異義語データ</option></select>`;
    panel.appendChild(selElem);
    const sel = panel.querySelector("#trainingData");
    const b0 = document.createElement("button");
    panel.appendChild(b0);
    b0.textContent = "学習データの生成";
    b0.addEventListener("click", e => {
        b0.disabled = true;
        setDatasets({ type: sel.value });
        b0.disabled = false;
    });
    if (datasetsMode === "default") {
        sel.value = "favorite";
        sel.style.display = "none";
    } else if (datasetsMode === "select") {
        b0.style.display = "none";
        sel.addEventListener("change", e => {
            console.log(e.target.value);
            b0.click();
        })
        b0.click();
    }

    // [className,label,initial_value,step_in_input]
    const params = [["epochs", "学習回数", 100, 1], ["learningRate", "学習率", 0.005, 0.001]];
    const b1 = document.createElement("button");
    b1.textContent = "学習";
    b1.addEventListener("click", async (e) => {
        b1.disabled = true;
        await learn({ datasets, ...params.map(e => e[0]).reduce((a, e) => ({ ...a, [e]: Number(panel.querySelector(`#${e}`).value) }), {}) });
        b1.disabled = false;
    });
    panel.appendChild(b1);
    params.forEach(([c, l, v, s]) => {
        const span = document.createElement("span");
        span.innerHTML = `<label for=${c}>${l}</label><input id=${c} type="number" value="${v}" step="${s}"/>`;
        panel.appendChild(span);
    })


    panel.className = "flex gap-x-2 items-center text-xs"
    panel.querySelectorAll("button").forEach(e => {
        e.className = "px-3 py-1 border-4 rounded hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent";
    });
    panel.querySelectorAll("input").forEach(e => {
        e.className = "inline-block w-16";
    });
    return learningPanelElem.container;
}
function setupModelEvaluationPanel({ predict }) {
    const elem = tfvis.visor().surface({ name: "モデルの検証" });
    let panel = elem.container.querySelector(".modelEvaluationPanel");
    if (panel === null) {
        panel = document.createElement("div");
        panel.className = "modelEvaluationPanel";
        panel.innerHTML = `<div><input id="prompt" type="text" placeholder="入力" /><button id="run">⏎</button><span class="message text-xs"></span></div>
                <table id="results">${models.map((m, i) => `<tr><td class="model">${m.model.name}</td><td class="out ${m.model.name}"></td></tr>`).join("")}</table>`;
        elem.container.appendChild(panel);

        const btn = panel.querySelector("button");
        const inp = panel.querySelector("input");
        btn.addEventListener("click", () => {
            predict({ models, datasets, input: inp.value });
        })
        inp.addEventListener("keydown", e => {
            if (e.key === "Enter") {
                btn.click();
            }
        })
    }
    panel.querySelectorAll(".message").forEach(e => e.textContent = "");
    panel.querySelectorAll(".out").forEach(e => e.textContent = "");

    return elem.container;
}
function updateModelEvaluationPanel({ modelEvaluationElem, results, errorMessage }) {
    if (results !== undefined) {
        Object.keys(results).forEach(name => {
            modelEvaluationElem.querySelector(`.out.${CSS.escape(name)}`).textContent = results[name];
        })
    }
    modelEvaluationElem.querySelector(`.message`).textContent = errorMessage ?? "";
}
function setupDatasetsPanel() {
    const datasetsElem = tfvis.visor().surface({ name: "学習データとテストデータ", styles: { height: 220 } });
    let panel = datasetsElem.container.querySelector(".datasetsPanel");
    if (panel === null) {
        panel = document.createElement("div");
        panel.className = "datasetsPanel flex gap-4";
        panel.innerHTML = '<div class="w-1/2"><span>学習データ</span><table class="training_data"></table></div><div class="w-1/2"><span>テストデータ</span><table class="test_data"></table></div>';
        datasetsElem.container.appendChild(panel);

        datasetsElem.container.querySelectorAll('table').forEach(e => { e.className += ' min-w-full border border-gray-300' })
    }
    return datasetsElem.container;
}
function updateDatasetsPanel({ datasetsElem, datasets }) {
    datasetsElem.querySelector(".training_data").innerHTML =
        "<tr><th>input</th><th>output</th></th>" +
        datasets.sentences.map(e => {
            const parts = e.split(" ");
            return `<tr><td>${parts.slice(0, parts.length - 1)}</td><td>${parts[parts.length - 1]}</td></tr>`;
        }).join("");
    if (datasets.test_patterns !== undefined && datasets.correct_answers !== undefined) {
        datasetsElem.querySelector(".test_data").innerHTML =
            "<tr><th>input</th><th>answer</th></th>" +
            datasets.test_patterns.map((e, i) => `<tr><td>${e}</td><td>${datasets.correct_answers[i]}</td></tr>`).join("")
    }
    datasetsElem.querySelectorAll('tr').forEach(e => { e.className += ' border px-3 py-2 text-left' })
    datasetsElem.querySelectorAll('td').forEach(e => { e.className += ' border text-xs' })
}
function setupResultsPanel({ tfvis, models, test_patterns, correct_answers, enableAttnScr = false }) {
    const resultsElem = tfvis.visor().surface({ name: "Results of Evaluation" });
    let tableElem = resultsElem.container.querySelector(".results_table");
    if (tableElem === null) {
        tableElem = document.createElement("table");
        tableElem.classList.add("results_table");
        resultsElem.container.appendChild(tableElem);

        const desc = '→横:影響語 / ↓縦:入力語、黒ほど高スコア (例: 0行1列が濃い=語0は語1に強く注目)';
        const pup = `<span class="absolute hidden group-hover:block bg-gray-800 text-white text-xs px-2 py-1 rounded">${desc}</span>`;
        tableElem.innerHTML = `<tr><th>model name</th><th>test data</th><th>correct answer</th><th>predicted</th>${enableAttnScr ? `<th class="group relative">attn scr${pup}</th>` : ""}</tr>`;

        models.forEach(e => {
            const tr = document.createElement("tr");
            tr.innerHTML += `<td>${e.model.name}</td>`;
            tr.innerHTML += `<td class="test_patterns"></td>`;
            tr.innerHTML += `<td class="correct_answers"></td>`;
            tr.innerHTML += `<td class="predicted ${e.model.name}_predicted"></td>`;

            if (enableAttnScr && e.options?.mha !== undefined) {
                const elem = document.createElement("canvas");
                elem.width = 128;
                elem.height = 110;
                elem.classList.add(`attnscr`);
                elem.classList.add(`${e.model.name}_attnscr`);
                tr.appendChild(elem);
            }
            tableElem.appendChild(tr);
        })

        resultsElem.container.querySelectorAll('table').forEach(e => { e.className += ' min-w-full border border-gray-300' })
        resultsElem.container.querySelectorAll('tr').forEach(e => { e.className += ' border px-3 py-2 text-left' })
        resultsElem.container.querySelectorAll('td').forEach(e => { e.className += ' border text-[10px]' })
    }
    resultsElem.container.querySelectorAll('.attnscr').forEach(e => { const c = e.getContext("2d"); c.clearRect(0, 0, c.canvas.width, c.canvas.height) });
    resultsElem.container.querySelectorAll('.predicted').forEach(e => e.innerHTML = "");
    if (test_patterns !== undefined) {
        resultsElem.container.querySelectorAll('.test_patterns').forEach(e => e.innerHTML = test_patterns.map(e => `<div class='cell'>${e}</div>`).join(""));
    }
    if (correct_answers !== undefined) {
        resultsElem.container.querySelectorAll('.correct_answers').forEach(e => e.innerHTML = correct_answers.map(e => `<div class='cell'>${e}</div>`).join(""));
    }
    resultsElem.container.querySelectorAll(".cell").forEach(e => e.className += " min-h-[1.5em] leading-[1.5em]");
    return resultsElem.container;
}
function updateResultsPanel({ modelEntry, datasets, resultsElem, results }) {
    if (resultsElem === undefined) {
        return;
    }
    // data [batch, numHeads, dim, dim] 0-1の間に正規化済
    function drawAttentions(ctx, data) {
        // console.log(data)
        const cellW = 8;
        const cellH = 8;
        const attrGap = 4;
        const attrWidth = data[0][0].length * cellW;

        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        data.forEach((attrs, iBatch) => {
            const offsetY = (attrWidth + attrGap) * iBatch;
            attrs.forEach((attr, mIndex) => {
                // console.log(Array.from(attr));
                const offsetX = (attrWidth + attrGap) * mIndex;
                attr.forEach((row, wIndex) => {
                    row.forEach((val, hInnerIndex) => {
                        const color = Math.floor((1 - val) * 255);
                        ctx.fillStyle = `rgb(${color},${color},${color})`;
                        ctx.fillRect(offsetX + wIndex * cellW, offsetY + hInnerIndex * cellH, cellW, cellH);
                    });
                });
            })
        });
    }
    const elem = resultsElem.querySelector(`.${CSS.escape(modelEntry.model.name)}_predicted`);
    elem.innerHTML = results.map(e => `<div class="cell"><span class="text-${e.correct_answer === e.predicted ? "blue" : "red"}-500">${e.predicted}</span></div>`).join("");
    elem.querySelectorAll(".cell").forEach(e => e.className += " min-h-[1.5em] leading-[1.5em]");

    const attnElem = resultsElem.querySelector(`.${CSS.escape(modelEntry.model.name)}_attnscr`)
    if (attnElem !== null && modelEntry.options?.mha !== undefined) {
        drawAttentions(attnElem.getContext("2d"), modelEntry.options.mha.getAttentionScores());
    }
}

// mode: next: [1,2,3,4]のデータがある場合、[1]から[2]、[1,2]から[3]、[1,2,3]から[4]というデータを作る
//       last: [1,2,3,4]の場合、常に最後だけを推測する[1,2,3]から[4]のデータを作る
function generateDatasets({ sentences, test_patterns, correct_answers, mode = "next" }) {
    const maxLen = Math.max(...sentences.map(e => e.split(" ").length));
    const allWords = [...new Set(sentences.join(" ").split(" "))].sort();
    const vocab = { "<PAD>": 0, ...allWords.reduce((a, e, i) => ({ ...a, [e]: i + 1 }), {}) }

    function encode(words) {
        const inputPad = new Array(maxLen - words.length - 1).fill(vocab["<PAD>"]);
        return [...inputPad, ...words.map(w => vocab[w])];
    }
    function toTensor(seq) {
        // disposeが必要、またはtidy内で実行する
        return tf.tensor2d(seq, [seq.length, maxLen - 1], 'int32');
    }
    function decode(code) {
        return allWords[code - 1];
    }
    function tokenize(input) {
        const words = Object.keys(vocab).sort((a, b) => b.length - a.length);
        const chunks = input.trim().split(/\s+/).filter(Boolean);
        const ids = [];
        for (const chunk of chunks) {
            let rest = chunk;
            while (rest.length > 0) {
                let matchedWord = null;
                for (const w of words) {
                    if (rest.startsWith(w)) {
                        matchedWord = w;
                        break;
                    }
                }
                if (!matchedWord) {
                    return { tokens: null, errorMessage: `未知の語が含まれています: "${rest}"（chunk="${chunk}"）` };
                }
                ids.push(vocab[matchedWord]);
                rest = rest.slice(matchedWord.length);
            }
        }
        return { tokens: Array(Math.max(0, maxLen - 1 - ids.length)).fill(0).concat(ids) };
    }

    const sequences = [];
    sentences.forEach(s => {
        const words = s.split(" ");
        const n = words.length;
        if (mode === "next") {
            for (let i = 1; i < n; i++) {
                sequences.push({ inputSeq: encode(words.slice(0, i)), targetWord: vocab[words[i]] });
            }
        } else if (mode === "last") {
            sequences.push({ inputSeq: encode(words.slice(0, n - 1)), targetWord: vocab[words[n - 1]] });
        }
    });

    const inputs = tf.tensor2d(sequences.map(e => e.inputSeq), [sequences.length, maxLen - 1], 'int32');
    const targets = tf.tensor1d(sequences.map(e => e.targetWord), 'float32');
    return { train_x: inputs, train_y: targets, maxLen, vocab, encode, toTensor, decode, sentences, sequences, test_patterns, correct_answers, tokenize };
}
function generateHomonymDatasets() {
    // 文脈: 順列を作って最後の語を予測する
    const contexts = [["道路", "車", "歩道", "交通"], ["食事", "食卓", "食器", "ご飯"]]
    // 文脈と"ハシ"を使った例題に含まれない語
    const noise = ["大", "量", "質", "間隔"];
    // ハシを使った例題 例: ["道路", "大", "ハシ", "わたる"], ["間隔", "食卓", "ハシ", "たべる"]
    let examples = contexts.flatMap((c, i) => {
        const ans = i === 0 ? "わたる" : "たべる";
        return c.flatMap(ci => noise.flatMap(e => [[ci, e, "ハシ", ans], [e, ci, "ハシ", ans]]))
    })
    const choice = (arr) => arr[Math.floor(Math.random() * arr.length)];
    const extract = (context_i, context_in_i, idx) => choice(examples.filter(e => e[idx] === contexts[context_i][context_in_i]));
    const [test_patterns, correct_answers] = (() => {
        const rec = [extract(0, 2, 0), extract(0, 3, 1), extract(1, 2, 0), extract(1, 3, 1)];
        return [rec.map(e => e.slice(0, e.length - 1)), rec.flatMap(e => e.slice(e.length - 1))]
    })();
    examples = examples.filter(e => !test_patterns.map(e => JSON.stringify(e)).includes(JSON.stringify(e.slice(0, e.length - 1))));

    const perm = a => a.length === 1 ? [a] : a.flatMap((e, i) => perm(a.filter((_, j) => j !== i)).map(t => [e, ...t]));
    const sentences = [];
    contexts.forEach(context => {
        sentences.push(...perm(context).map(e => e.join(" ")))
    })
    examples.forEach(example => {
        sentences.push(example.join(" "))
    })

    return generateDatasets({ sentences, test_patterns, correct_answers, mode: "last" })
}


function evaluateModel({ model, datasets }) {
    return tf.tidy(() => {
        const inp = datasets.toTensor(datasets.test_patterns.map(e => datasets.encode(e)))
        const probs = model.predict(inp)
        const predIds = probs.argMax(-1).dataSync();
        const predWords = Array.from(predIds).map((e, i) => datasets.decode(e));
        console.log(model.name + "\n" + predWords.map((e, i) => `${datasets.test_patterns[i]} ${e}`).join("\n"));
        return datasets.test_patterns.map((e, i) => ({ test_pattern: datasets.test_patterns[i], predicted: predWords[i], correct_answer: datasets.correct_answers?.[i] }));
    })
}

function setDatasets({ type = "favorite" } = {}) {
    // generateFavoriteDatasets, generateHomonymDatasets

    if (type === "favorite") {
        datasets = generateFavoriteDatasets();
    } else if (type === "homonym") {
        datasets = generateHomonymDatasets();
    }

    const datasetsElem = setupDatasetsPanel();
    updateDatasetsPanel({ datasetsElem, datasets })
    return datasets;
}
function setModels({ learningRate = 0.001, verbose = false } = {}) {
    const keyDim = 8;
    const numHeads = 1;
    models = [
        createSimpleLLM({
            vocabSize: Object.keys(datasets.vocab).length,
            inputDim: datasets.train_x.shape[1],
            keyDim,
            numHeads,
            learningRate,
        }),
        createSimpleGAP({
            vocabSize: Object.keys(datasets.vocab).length,
            inputDim: datasets.train_x.shape[1],
            keyDim,
            learningRate,
            type: "ful",
        }),
        // createSimpleGAP({
        //     vocabSize: Object.keys(datasets.vocab).length,
        //     inputDim: datasets.train_x.shape[1],
        //     keyDim,
        //     learningRate,
        //     type: "slc",
        // }),
        createSimpleFNN({
            vocabSize: Object.keys(datasets.vocab).length,
            inputDim: datasets.train_x.shape[1],
            keyDim,
            learningRate,
            type: "ful",
        }),
        // createSimpleFNN({
        //     vocabSize: Object.keys(datasets.vocab).length,
        //     inputDim: datasets.train_x.shape[1],
        //     keyDim,
        //     learningRate,
        //     type: "slc",
        // }),
    ].filter(e => e !== undefined);
    if (verbose) {
        models.forEach(e => {
            tfvis.show.modelSummary({ name: e.model.name }, e.model);
        })
    }
    return models;
}
async function learn({ datasets, learningRate, epochs, verbose = true }) {
    if (datasets === undefined) {
        alert("学習データを生成してください。")
        return;
    }
    setModels({ learningRate });
    const resultsElem = datasets.test_patterns ? setupResultsPanel({ tfvis, models, test_patterns: datasets.test_patterns, correct_answers: datasets.correct_answers }) : undefined;

    for (let i = 0; i < models.length; i += 1) {
        const model = models[i].model;
        models[i].options?.mha?.setKeepAttentionScores(false);
        const history = await model.fit(datasets.train_x, datasets.train_y, {
            epochs,
            batchSize: 8,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: "学習回数と誤差" },
                ["loss"],
                { height: 80, callbacks: ["onEpochEnd"] },
            ),
        });
        models[i].options?.mha?.setKeepAttentionScores(true);
        if (datasets.test_patterns !== undefined) {
            const results = evaluateModel({ model, datasets });
            updateResultsPanel({ modelEntry: models[i], datasets, resultsElem, results })
        }
    }
    function predict({ models, datasets, input }) {
        return tf.tidy(() => {
            const { tokens, errorMessage } = datasets.tokenize(input);
            if (tokens === null) {
                updateModelEvaluationPanel({ modelEvaluationElem, errorMessage });
                return;
            } else if (tokens.length > datasets.maxLen - 1) {
                updateModelEvaluationPanel({ modelEvaluationElem, errorMessage: `語数が多いです。最大 ${datasets.maxLen - 1}` });
                return;
            }
            const x = datasets.toTensor([tokens]);
            // console.log(x)
            const results = {};
            models.map(e => e.model).forEach(model => {
                const probs = model.predict(x)
                const predIds = probs.argMax(-1).dataSync();
                const predWords = Array.from(predIds).map((e, i) => datasets.decode(e));
                console.log(model.name + "\n" + predWords.map((e, i) => `${input} ${e}`).join("\n"));

                results[model.name] = predWords.join(" ");
            })
            updateModelEvaluationPanel({ modelEvaluationElem, results })
        })
    }
    const modelEvaluationElem = setupModelEvaluationPanel({ predict });
}
