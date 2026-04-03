import asyncio
import chess
import chess.engine
import chess.syzygy
import math
import json
import time
import statistics

# ---------------------------------------------------------
# 1. 정규화 레이어 (Sigmoid 기반 - 추후 엔진별 보정 예정)
# ---------------------------------------------------------
def normalize_to_win_prob(score: chess.engine.Score) -> float:
    val = score.pov(chess.WHITE).score(mate_score=10000)
    if val is None:
        return 0.5
    return 1 / (1 + math.pow(10, -val / 400))

# ---------------------------------------------------------
# 2. 엔진 워커 (Fan-out 대상 - Multi-PV 적용)
# ---------------------------------------------------------
async def analyze_position(engine, engine_name, board, limit, multipv=3):
    """각 엔진이 Top-K(Multi-PV) 후보 수를 분석하여 반환"""
    start_time = time.time()
    
    # multipv 인자를 주면 List[InfoDict] 형태로 여러 라인을 반환함
    infos = await engine.analyse(board, limit, multipv=multipv, info=chess.engine.INFO_ALL)
    
    # 일부 구버전 엔진이나 설정 문제로 단일 dict가 올 경우 처리
    if isinstance(infos, dict):
        infos = [infos]
        
    parsed_lines = []
    for info in infos:
        if "pv" in info and info["pv"]:
            parsed_lines.append({
                "move": info["pv"][0].uci(),
                "score": info["score"],
                "win_prob": normalize_to_win_prob(info["score"]),
                "depth": info.get("depth", 0)
            })
            
    elapsed = time.time() - start_time
    
    # 대표 메타데이터(총 노드 수, 속도 등)는 첫 번째 라인(Top 1)에서 추출
    top_info = infos[0] if infos else {}
    
    return {
        "engine_name": engine_name,
        "lines": parsed_lines, # Top-K 후보 수 배열
        "nodes": top_info.get("nodes", 0),
        "nps": top_info.get("nps", 0),
        "time_used": elapsed
    }

# ---------------------------------------------------------
# 3. 중앙 집계기 (Pool 기반 가중합 & 예외 신호 감지)
# ---------------------------------------------------------
def aggregate_results(results, weights):
    aggregated_data = {
        "exception_triggered": False,
        "bypass_reasons": [],
        "final_move": None,
        "final_win_prob": 0.5,
        "engine_details": results
    }
    
    pool = {}           # { "e2e4": 최종 가중합 점수 }
    top1_probs = []     # 엔진 간 분산(Variance) 계산용
    
    for res in results:
        engine_name = res["engine_name"]
        weight = weights.get(engine_name, 1.0)
        lines = res["lines"]
        
        if not lines:
            continue
            
        top1_prob = lines[0]["win_prob"]
        top1_probs.append(top1_prob)
        
        # [Signal 1] Mate 또는 극단적 승률 감지
        if lines[0]["score"].is_mate() or top1_prob > 0.95 or top1_prob < 0.05:
            aggregated_data["exception_triggered"] = True
            aggregated_data["bypass_reasons"].append(f"Extreme Score/Mate by {engine_name}")
            
        # [Signal 2] Top1 - Top2 Gap 확인 (불확실성 감지)
        if len(lines) >= 2:
            gap = abs(lines[0]["win_prob"] - lines[1]["win_prob"])
            if gap < 0.02:  # 1순위와 2순위 승률 차이가 2% 미만이면 매우 불안정함
                aggregated_data["exception_triggered"] = True
                aggregated_data["bypass_reasons"].append(f"Low Top1-Top2 Gap ({gap:.3f}) in {engine_name}")
                
        # Move 풀링 및 가중합 계산
        for rank, line in enumerate(lines):
            move = line["move"]
            prob = line["win_prob"]
            
            # 순위(Rank)에 따른 약간의 페널티 부여도 가능하지만, 일단 확률*가중치만 적용
            if move not in pool:
                pool[move] = 0.0
            pool[move] += prob * weight

    # [Signal 3] 엔진 간 의견 분산(Variance) 감지
    if len(top1_probs) >= 2:
        variance = statistics.variance(top1_probs)
        if variance > 0.05: # 분산 임계값 (조정 가능)
            aggregated_data["exception_triggered"] = True
            aggregated_data["bypass_reasons"].append(f"High Engine Variance ({variance:.3f})")

    if pool:
        # 풀(Pool)에서 가장 높은 가중합을 받은 수 선택
        best_move = max(pool, key=pool.get)
        aggregated_data["final_move"] = best_move
        
        # 승률 평균치 (간단한 정규화)
        total_weight = sum(weights.values())
        aggregated_data["final_win_prob"] = pool[best_move] / total_weight if total_weight > 0 else 0.5
        
    return aggregated_data

# ---------------------------------------------------------
# 4. 오케스트레이터 (TB 인터셉트 포함)
# ---------------------------------------------------------
async def run_benchmark(fens, engine_paths, tb_path, limit, weights, multipv=3):
    # 엔진 및 테이블베이스 초기화
    engines = {}
    for name, path in engine_paths.items():
        _, engine = await chess.engine.popen_uci(path)
        engines[name] = engine
        
    # Syzygy 테이블베이스 로드 (경로가 유효할 경우)
    tb = None
    try:
        tb = chess.syzygy.open_tablebase(tb_path)
        print("[System] Syzygy Tablebase 로드 성공.")
    except Exception as e:
        print(f"[System] Tablebase 없이 진행합니다. (사유: {e})")

    training_data_log = []
    
    try:
        for idx, fen in enumerate(fens):
            print(f"[{idx+1}/{len(fens)}] FEN 분석 중: {fen}")
            board = chess.Board(fen)
            
            # [최우선] Tablebase 인터셉트 (기물 수 7개 이하)
            if tb and len(board.piece_map()) <= 7:
                wdl = tb.probe_wdl(board)
                dtz = tb.probe_dtz(board)
                
                # TB에서 승패가 확정된 경우 (0은 무승부, 양수는 승리, 음수는 패배)
                if wdl is not None:
                    print(f" -> [TB Intercept] 완벽한 엔드게임 해답 발견! (WDL: {wdl}, DTZ: {dtz})")
                    # 실전에서는 여기서 합법적인 수(Legal moves) 중 DTZ를 줄이는 최선의 수를 찾아 바로 반환
                    continue # 엔진 분석 스킵

            # Fan-out: 여러 엔진에 동시에 분석 요청 (Multi-PV)
            tasks = [
                analyze_position(engine, name, board, limit, multipv)
                for name, engine in engines.items()
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Fan-in & 집계
            final_result = aggregate_results(results, weights)
            
            # 로깅
            log_entry = {
                "fen": fen,
                "final_selection": final_result["final_move"],
                "final_win_prob": final_result["final_win_prob"],
                "is_exception": final_result["exception_triggered"],
                "bypass_reasons": final_result["bypass_reasons"],
                "engine_evals": results
            }
            training_data_log.append(log_entry)
            
            reasons = f" | 예외: {final_result['bypass_reasons']}" if final_result["exception_triggered"] else ""
            print(f" -> 최종 선택: {log_entry['final_selection']}{reasons}")
            
    finally:
        for engine in engines.values():
            await engine.quit()
        if tb:
            tb.close()
            
    with open("ensemble_training_data.json", "w") as f:
        json.dump(training_data_log, f, indent=4)
        
    print("\n[System] 벤치마크 완료! 'ensemble_training_data.json' 확인 요망.")

# ---------------------------------------------------------
# 실행 영역
# ---------------------------------------------------------
# if __name__ == "__main__":
#     test_fens = [
#         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # 오프닝
#         "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"                           # TB 인터셉트 테스트 (킹+폰 엔드게임)
#     ]
#     
#     engine_paths = {"Stockfish_18": "./stockfish18", "Komodo": "./komodo"}
#     tb_path = "./syzygy" # 3-4-5-6-7 man Syzygy 폴더 경로
#     
#     ensemble_weights = {"Stockfish_18": 1.2, "Komodo": 0.8}
#     search_limit = chess.engine.Limit(nodes=100000) 
#     
#     # asyncio.run(run_benchmark(test_fens, engine_paths, tb_path, search_limit, ensemble_weights, multipv=3))
