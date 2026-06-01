"""
RAGAS 评估脚本 —— 量化评估 RAG 系统的检索和生成质量

用法：python evaluation/eval_rag.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_deepseek import ChatDeepSeek

import config_data as config
from rag import RagService


def load_test_cases(filepath: str) -> list[dict]:
    """加载测试数据集"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(test_cases: list[dict]):
    rag = RagService()

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    session_config = {"configurable": {"session_id": "eval_session"}}

    for i, case in enumerate(test_cases):
        question = case["question"]
        questions.append(question)
        ground_truths.append(case["ground_truth"])

        # 生成答案
        try:
            result = rag.chain.invoke({"input": question}, session_config)
            answers.append(result)
        except Exception as e:
            print(f"  [{i+1}] 生成失败: {e}")
            answers.append("")

        # 获取检索上下文
        try:
            docs = rag.search_service.search(question)
            contexts_list.append([doc.page_content for doc in docs])
        except Exception as e:
            print(f"  [{i+1}] 检索失败: {e}")
            contexts_list.append([])

        print(f"  [{i+1}/{len(test_cases)}] {question[:40]}... ✓")

    # 构建评估数据集
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # 用 DeepSeek 作为评判 LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatDeepSeek(model=config.chat_model)
    )

    # 用 BGE 作为评估用的 Embedding（替代默认的 OpenAI）
    eval_embeddings = LangchainEmbeddingsWrapper(config.embeddings)

    # 运行评估
    print(f"\n正在评估 {len(test_cases)} 条测试用例...")
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=eval_embeddings,
    )

    return result


def print_results(result):
    df = result.to_pandas()

    print("\n" + "=" * 60)
    print("  RAG 系统评估报告（RAGAS）")
    print("=" * 60)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = {
        "faithfulness":        "忠实度 (Faithfulness)     ",
        "answer_relevancy":    "答案相关性 (Answer Relevancy)",
        "context_precision":   "上下文精度 (Context Precision)",
        "context_recall":      "上下文召回 (Context Recall) ",
    }

    for col in metrics:
        if col in df.columns:
            mean_val = df[col].mean()
            bar = "█" * int(mean_val * 40) + "░" * (40 - int(mean_val * 40))
            print(f"  {labels.get(col, col)}: {mean_val:.4f}  {bar}")

    print("-" * 60)
    print(f"  测试用例数: {len(df)}")
    print(f"  检索文档数 (top_k): {config.similarity_top_k}")

    # RAGAS 结果中答案列名是 "response"，检索上下文列名是 "retrieved_contexts"
    ans_col = "response" if "response" in df.columns else "answer"
    non_empty = sum(1 for a in df.get(ans_col, []) if a and a.strip())
    print(f"  有效答案数: {non_empty}/{len(df)}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_dataset.json"
    )
    test_cases = load_test_cases(dataset_path)
    print(f"已加载 {len(test_cases)} 条测试用例\n")

    result = run_evaluation(test_cases)
    df = print_results(result)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n详细结果已保存至: {output_path}")
