# ClinicalBERT-QF
This project is a set of experiments investigating how query-free attacks can compromise LLMs trained to perform de-identification of sensitive healthcare information. Zhuang et al have demonstrated that text-to-image models like StableDiffusion are susceptible (https://github.com/OPTML-Group/QF-Attack/). This project applies similar techniques to simulate attacks against masked language models like ClinicalBERT.

<div align="center">
  <img src="/images/homoglyph.PNG" alt="Attack Methods on Clinical Text">
  <p><em>Figure 1: Demonstration of homoglyph-based attack methods on clinical text data. Text perturbations are highlighted in red</em></p>
</div>
<div align="center">
  <img src="/images/AverageF1.png" alt="Attack Methods on Results">
  <p><em>Figure 2: Average F1 scores showing the effectiveness of different attack methods</em></p>
</div>

