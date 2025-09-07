# ElecDevQA
The ElecDevQA dataset is a valuable, ground-truth resource for question-answering tasks within the electronic device domain. Each question-and-answer pair was curated using a hybrid Retrieval-Augmented Generation (RAG) approach that combines information from a structured knowledge graph (derived from IEEE abstracts) with relevant content from research articles and web pages. This methodology ensures the dataset's answers are not only accurate but also rich in context, as they are grounded in multiple credible sources. Consequently, ElecDevQA serves as an essential benchmark for evaluating the domain-specific knowledge of AI models and for fine-tuning Large Language Models (LLMs) to provide more sophisticated and reliable answers to specialized queries.
## Question types
The questions within the ElecDevQA dataset are categorized into six distinct types, based on the specific information they are designed to query:
- FUNCTION: The device's primary function.
- FEATURE: A special characteristic of the device.
- PROPERTY: A measurable property of the device.
- COMPONENT: A part or component used in the device.
- MATERIAL: The material used in the device.
- TECHNOLOGY: The technology the device uses.

This categorization allows for a fine-grained analysis of model performance on different aspects of electronic device knowledge.
## Data Format
ElecDevQA JSON format data is structured as:
```json
{
"question": "What semiconductor materials are commonly found in Monolithic Microwave Integrated Circuit (MMIC) Doherty power amplifiers?",
"answer": "MMIC Doherty power amplifiers primarily use gallium arsenide (GaAs) or gallium nitride (GaN) for their high power density and efficiency. These devices are often fabricated on substrates like high-resistivity silicon or sapphire to minimize losses.  Other materials like indium phosphide (InP) and silicon germanium (SiGe) are also explored for specific performance benefits.",
"references": [
    "http://ieeexplore.ieee.org/document/6997915/",
    "https://en.wikipedia.org/wiki/Monolithic_microwave_integrated_circuit",
    "https://www.mdpi.com/2072-666X/5/3/711",
    "https://parts.jpl.nasa.gov/mmic/3-IX.PDF"
],
"question_type": "material"
}
