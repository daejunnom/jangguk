import asyncio
import chess
import chess.engine
import math
import json
import time

# ---------------------------------------------------------
# 1. 정규화 레이어 (Normalization Layer)
# ---------------------------------------------------------
def normalize_to_win_prob(score: chess.engine.Score) -> float:
    """cp 또는 mate 점수를 0.0 ~ 1.0 사이의 승률(Win Probability)로 변환"""
    # 백(White) 기준의 점수로 통일 (mate 점수는 ±10000 cp로 치환)
    val = score.pov(chess.WHITE).score(mate_score=10000)
    
    if val is None:
        return 0.5 # 평가 불가 시 0.5
        
    # WDL Sigmoid 변환 공식 (Stockfish 모델 참고)
    # P(win) = 1 / (1 + 10^(-cp / 400))
    win_prob = 1 / (1 + math.pow(10, -val / 400))
    return win_prob

# ---------------------------------------------------------
# 2. 엔진 워커 (Fan-out 대상)
# ---------------------------------------------------------
async def analyze_position(engine, engine_name, board, limit):
    """개별 엔진이 포지션을 분석 (play 대신 analyse 사용)"""
    start_time = time.time()
    
    # info=chess.engine.INFO_ALL 을 통해 cp, mate, pv 등 모든 정보 수집
    info = await engine.analyse(board, limit, info=chess.engine.INFO_ALL)
    
    elapsed = time.time() - start_time
    
    return {
        "engine_name": engine_name,
        "best_move": info.get("pv")[0].uci() if info.get("pv") else None,
        "score": info.get("score"),
        "win_prob": normalize_to_win_prob(info.get("score")),
        "nodes": info.get("nodes", 0),
        "nps": info.get("nps", 0),
        "depth": info.get("depth", 0),
        "time_used": elapsed
    }

# ---------------------------------------------------------
# 3. 중앙 집계기 (Aggregator / Fan-in 계층)
# ---------------------------------------------------------
def aggregate_results(results, weights):
    """수집된 결과를 바탕으로 최종 수 결정 및 예외 처리"""
    aggregated_data = {
        "exception_triggered": False,
        "final_move": None,
        "final_win_prob": 0.0,
        "engine_details": results
    }
    
    # [1] 예외 통로 (Bypass / Lone Wolf Signal)
    # 누군가 Mate를 발견했거나, 95% 이상의 극단적 승률을 보장한다면 즉시 채택
    for res in results:
        if res["score"].is_mate() or res["win_prob"] > 0.95 or res["win_prob"] < 0.05:
            aggregated_data["exception_triggered"] = True
            aggregated_data["final_move"] = res["best_move"]
            aggregated_data["final_win_prob"] = res["win_prob"]
            aggregated_data["bypass_reason"] = f"Critical signal from {res['engine_name']}"
            return aggregated_data
            
    # [2] 가중합 계산 (Weighted Sum)
    move_scores = {}
    total_prob = 0.0
    
    for res in results:
        move = res["best_move"]
        weight = weights.get(res["engine_name"], 1.0)
        
        if move not in move_scores:
            move_scores[move] = 0.0
            
        # 선택한 수에 승률 * 가중치를 더함 (Policy + Value 결합 모방)
        move_scores[move] += res["win_prob"] * weight
        total_prob += res["win_prob"] * weight
        
    # 가장 높은 가중합을 받은 수 선택
    best_move = max(move_scores, key=move_scores.get)
    aggregated_data["final_move"] = best_move
    
    # 승률 평균치 (간단한 정규화)
    total_weight = sum(weights.values())
    aggregated_data["final_win_prob"] = total_prob / total_weight if total_weight > 0 else 0.5
    
    return aggregated_data

# ---------------------------------------------------------
# 4. 오케스트레이터 및 벤치마크 하네스
# ---------------------------------------------------------
async def run_benchmark(fens, engine_paths, limit, weights):
    # 엔진 인스턴스 초기화 (프로세스 유지)
    engines = {}
    for name, path in engine_paths.items():
        _, engine = await chess.engine.popen_uci(path)
        engines[name] = engine
        
    training_data_log = []
    
    try:
        for idx, fen in enumerate(fens):
            print(f"[{idx+1}/{len(fens)}] FEN 분석 중: {fen}")
            board = chess.Board(fen)
            
            # Fan-out: 여러 엔진에 동시에 분석 요청
            tasks = [
                analyze_position(engine, name, board, limit)
                for name, engine in engines.items()
            ]
            
            # Fan-in: 모든 엔진의 분석이 끝날 때까지 대기 후 수집
            results = await asyncio.gather(*tasks)
            
            # 중앙 집계
            final_result = aggregate_results(results, weights)
            
            # 5. ML 학습 데이터 포맷 로깅
            log_entry = {
                "fen": fen,
                "final_selection": final_result["final_move"],
                "final_win_prob": final_result["final_win_prob"],
                "is_exception": final_result["exception_triggered"],
                "engine_evals": [
                    {
                        "engine": r["engine_name"],
                        "move": r["best_move"],
                        "win_prob": r["win_prob"],
                        "nodes": r["nodes"],
                        "time_used": round(r["time_used"], 3)
                    } for r in results
                ]
            }
            training_data_log.append(log_entry)
            
            # 진행 상황 출력
            print(f" -> 최종 선택: {log_entry['final_selection']} (Exception: {log_entry['is_exception']})")
            
    finally:
        # 워커 종료
        for engine in engines.values():
            await engine.quit()
            
    # 학습 데이터 JSON 저장
    with open("ensemble_training_data.json", "w") as f:
        json.dump(training_data_log, f, indent=4)
        
    print("\n[System] 벤치마크 완료! 데이터가 'ensemble_training_data.json'에 저장되었습니다.")

# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------
# if __name__ == "__main__":
#     test_fens = [
#         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # 시작
#         "5r1k/p1q2p2/bpp1pNpb/4Pn2/2p1R1QN/3P3P w - - 0 1"         # 전술
#     ]
#     
#     # Colab 경로에 맞게 수정 필요
#     engine_paths = {
#         "Stockfish_18": "./stockfish18",
#         "Komodo": "./komodo"
#     }
#     
#     # 엔진별 가중치 (향후 ML 모델이 동적으로 할당할 부분)
#     ensemble_weights = {"Stockfish_18": 1.2, "Komodo": 0.8}
#     
#     # 노드 제한 (시간이 아닌 노드로 제한하여 안정성 확보)
#     search_limit = chess.engine.Limit(nodes=500000) 
#     
#     # Colab 셀에서는 await run_benchmark(...) 직접 실행
#     asyncio.run(run_benchmark(test_fens, engine_paths, search_limit, ensemble_weights))