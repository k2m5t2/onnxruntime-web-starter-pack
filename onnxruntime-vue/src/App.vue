<script setup>
import HelloWorld from './components/HelloWorld.vue'
import TheWelcome from './components/TheWelcome.vue'
import { Tensor, InferenceSession } from "onnxruntime-web";
import { ref } from 'vue';


let session = ref(0);
let result = ref(0);

async function initiate_ort() {
  const url = "./model.onnx";
  // create a new session and load the specific model.
  session.value = await InferenceSession.create(url);
}

async function compute_something() {
    try {
        // prepare inputs. a tensor need its corresponding TypedArray as data
        const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        const tensorA = new Tensor('float32', dataA, [3, 4]);
        const tensorB = new Tensor('float32', dataB, [4, 3]);

        // prepare feeds. use model input names as keys.
        const feeds = { a: tensorA, b: tensorB };

        // feed inputs and run
        const results = await session.value.run(feeds);

        // read from results
        const dataC = results.c.data;
        console.log(`data of result tensor 'c': ${dataC}`);

        result.value = dataC;

    } catch (e) {
        console.log(`failed to inference ONNX model: ${e}.`);
    }
}

</script>

<template>
  <!-- <header>
    <img alt="Vue logo" class="logo" src="./assets/logo.svg" width="125" height="125" />

    <div class="wrapper">
      <HelloWorld msg="You did it!" />
    </div>
  </header> -->

  <!-- <main>
    <TheWelcome />
  </main> -->

  <div>
    <p>Result: {{ result }}</p>
    <button @click="initiate_ort">Initiate ONNX Runtime</button>
    <button @click="compute_something">Run a computation in ORT</button>
  </div>

</template>

<style scoped>
header {
  line-height: 1.5;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }
}
</style>
