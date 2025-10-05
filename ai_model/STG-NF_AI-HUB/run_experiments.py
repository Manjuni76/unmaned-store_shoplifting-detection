import subprocess
import sys
import time
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def run_experiment(subset_name, seed=42):
    """특정 관절 부위와 시드로 실험 실행"""
    print(f"🚀 Running {subset_name} (seed: {seed})...", end=" ", flush=True)
    
    # 관절 부위에 따라 배치 크기 최적화
    batch_size = 256
    epochs = 50
    
    try:
        # subprocess로 train_eval.py 실행하고 입력 자동화
        process = subprocess.Popen(
            [sys.executable, 'train_eval.py', 
             '--batch_size', str(batch_size), 
             '--epochs', str(epochs)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr을 stdout으로 리디렉션
            text=True,
            universal_newlines=True
        )
        
        # 관절 부위와 시드 선택 입력 (여러 줄 입력)
        inputs = f"{subset_name}\n{seed}\n"
        process.stdin.write(inputs)
        process.stdin.close()
        
        print(f"\n⏳ 학습 진행 중 (배치 크기: {batch_size}, 에포크: {epochs})...")
        
        # 실시간 출력 캡처
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())  # 실시간 출력
                output_lines.append(line)
        
        stdout = ''.join(output_lines)
        stderr = ""  # stderr은 이미 stdout에 포함됨
        
        # 결과만 추출
        results = extract_results(stdout, stderr)  # stderr도 전달
        
        if process.returncode == 0 and results:
            print("✅")
            return True, results
        else:
            print("❌")
            if stderr:
                print(f"Error: {stderr[-200:]}")  # 마지막 200자만 표시
            return False, None
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        process.kill()
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None

def extract_results(output, stderr_output=""):
    """출력에서 결과값과 학습 곡선 데이터 추출"""
    results = {}
    
    try:
        # AUC(ROC) 추출
        auc_roc_match = re.search(r'auc\(roc\):\s*([\d.]+)', output)
        if auc_roc_match:
            results['auc_roc'] = float(auc_roc_match.group(1))
        
        # AUC(PR) 추출  
        auc_pr_match = re.search(r'auc\(pr\):\s*([\d.]+)', output)
        if auc_pr_match:
            results['auc_pr'] = float(auc_pr_match.group(1))
        
        # EER 추출
        eer_match = re.search(r'eer:\s*([\d.]+)', output)
        if eer_match:
            results['eer'] = float(eer_match.group(1))
        
        # EER threshold 추출
        eer_threshold_match = re.search(r'eer threshold:\s*([\d.-]+)', output)
        if eer_threshold_match:
            results['eer_threshold'] = float(eer_threshold_match.group(1))
        
        # 샘플 수 추출
        samples_match = re.search(r'Number of samples\s*(\d+)', output)
        if samples_match:
            results['num_samples'] = int(samples_match.group(1))
        
        # Learning rate 데이터 추출 (stdout + stderr 둘 다 확인)
        learning_data = extract_learning_curves(output, stderr_output)
        if learning_data:
            results['learning_curves'] = learning_data
            
        return results if results else None
        
    except Exception as e:
        print(f"결과 추출 오류: {e}")
        return None

def extract_learning_curves(stdout_output, stderr_output=""):
    """학습 곡선 데이터 추출 (loss, learning rate 등)"""
    learning_data = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    try:
        # Learning rate 추출 (stdout에서)
        lr_patterns = [
            r'New LR:\s*([\d.e-]+)',
            r'lr:\s*([\d.e-]+)',
            r'learning_rate:\s*([\d.e-]+)',
            r'LR:\s*([\d.e-]+)'
        ]
        
        for pattern in lr_patterns:
            matches = re.findall(pattern, stdout_output)
            if matches:
                learning_data['learning_rates'] = [float(lr) for lr in matches]
                break
        
        # Loss 값들을 stderr에서 추출 (진행률 바와 함께 출력됨)
        if stderr_output:
            # 각 에포크의 마지막 Loss 값을 추출 (에포크 끝에 나오는 값)
            loss_pattern = r'Loss:\s*([\d.]+):\s*100%.*?(\d+)/(\d+)'
            epoch_loss_matches = re.findall(loss_pattern, stderr_output)
            
            if epoch_loss_matches:
                epoch = 1
                for loss_str, current, total in epoch_loss_matches:
                    learning_data['epochs'].append(epoch)
                    learning_data['train_loss'].append(float(loss_str))
                    epoch += 1
            else:
                # 백업: 모든 Loss 값 중에서 대표값들을 추출
                all_loss_matches = re.findall(r'Loss:\s*([\d.]+)', stderr_output)
                if all_loss_matches:
                    # 에포크별로 나누어서 각 에포크의 마지막 loss 추출
                    # stderr 출력을 에포크별로 분할
                    epoch_sections = re.split(r'(\d+)%\|#+\|\s*\d+/\d+.*?\[.*?\]', stderr_output)
                    
                    # 각 섹션에서 마지막 Loss 값 추출 (간단한 방법)
                    loss_values = []
                    prev_loss = None
                    for loss_str in all_loss_matches:
                        current_loss = float(loss_str)
                        if prev_loss is None or abs(current_loss - prev_loss) > 0.1:
                            # 의미있는 변화가 있을 때만 추가
                            if len(loss_values) < 10:  # 최대 10개 에포크까지만
                                loss_values.append(current_loss)
                                prev_loss = current_loss
                    
                    # 에포크별로 대략적인 대표값들 선택
                    if len(loss_values) >= 3:
                        step = len(loss_values) // 3
                        selected_losses = [loss_values[0], loss_values[step], loss_values[-1]]
                        
                        for i, loss in enumerate(selected_losses):
                            learning_data['epochs'].append(i + 1)
                            learning_data['train_loss'].append(loss)
        
        print(f"[DEBUG] 추출된 학습 데이터: epochs={len(learning_data['epochs'])}, train_loss={len(learning_data['train_loss'])}, lr={len(learning_data['learning_rates'])}")
        if learning_data['train_loss']:
            print(f"[DEBUG] Loss 값들: {learning_data['train_loss']}")
        
        # 데이터가 있으면 반환, 없으면 None 반환
        if learning_data['epochs'] or learning_data['learning_rates']:
            return learning_data
        
        return None
        
    except Exception as e:
        print(f"학습 곡선 데이터 추출 오류: {e}")
        return None

def save_learning_curves_plot(all_results, time_str):
    """학습 곡선 그래프를 PNG 파일로 저장"""
    try:
        # 결과 디렉토리 확인
        plots_dir = os.path.join("results", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 각 관절 부위별로 그래프 생성
        for subset, seed_results in all_results.items():
            if not seed_results:
                continue
                
            # 해당 부위에 학습 곡선 데이터가 있는지 확인
            learning_curves_data = []
            for seed, results in seed_results.items():
                if results and 'learning_curves' in results:
                    learning_curves_data.append((seed, results['learning_curves']))
            
            if not learning_curves_data:
                print(f"📊 {subset} 부위에 학습 곡선 데이터가 없습니다.")
                continue
            
            # 그래프 생성
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Learning Curves - {subset.upper()}', fontsize=16)
            
            # 각 시드별 데이터 플롯
            colors = plt.cm.tab10(np.linspace(0, 1, len(learning_curves_data)))
            
            has_train_loss = False
            has_val_loss = False
            has_learning_rate = False
            
            for i, (seed, learning_data) in enumerate(learning_curves_data):
                color = colors[i]
                label = f'Seed {seed}'
                
                # Training Loss
                if learning_data['epochs'] and learning_data['train_loss']:
                    axes[0, 0].plot(learning_data['epochs'], learning_data['train_loss'], 
                                   color=color, label=label, marker='o', markersize=4, linewidth=2)
                    has_train_loss = True
                
                # Validation Loss (있는 경우)
                if learning_data['epochs'] and learning_data['val_loss']:
                    axes[0, 1].plot(learning_data['epochs'], learning_data['val_loss'], 
                                   color=color, label=label, marker='s', markersize=4, linewidth=2)
                    has_val_loss = True
                
                # Learning Rate
                if learning_data['learning_rates']:
                    epochs_lr = list(range(1, len(learning_data['learning_rates']) + 1))
                    
                    # 시드별로 다른 스타일 적용 (같은 값이어도 구분 가능하도록)
                    linestyles = ['-', '--', '-.', ':', '-']
                    markers = ['o', 's', '^', 'D', 'v']
                    linestyle = linestyles[i % len(linestyles)]
                    marker = markers[i % len(markers)]
                    
                    axes[1, 0].plot(epochs_lr, learning_data['learning_rates'], 
                                   color=color, label=label, linestyle=linestyle,
                                   marker=marker, markersize=6, linewidth=2, alpha=0.8)
                    axes[1, 0].set_yscale('log')
                    has_learning_rate = True
                
                # Combined Loss Comparison
                if learning_data['epochs'] and learning_data['train_loss']:
                    axes[1, 1].plot(learning_data['epochs'], learning_data['train_loss'], 
                                   color=color, label=f'{label} (Train)', linestyle='-', marker='o', markersize=3, linewidth=2)
                if learning_data['epochs'] and learning_data['val_loss']:
                    axes[1, 1].plot(learning_data['epochs'], learning_data['val_loss'], 
                                   color=color, label=f'{label} (Val)', linestyle='--', marker='s', markersize=3, linewidth=1.5)
            
            # 축 설정
            axes[0, 0].set_title('Training Loss by Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            if has_train_loss:
                axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].set_title('Validation Loss by Epoch')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            if has_val_loss:
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No Validation Loss Data', 
                               ha='center', va='center', transform=axes[0, 1].transAxes, 
                               fontsize=12, alpha=0.7)
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate (log scale)')
            if has_learning_rate:
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No Learning Rate Data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes, 
                               fontsize=12, alpha=0.7)
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].set_title('Training Loss Comparison')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            if has_train_loss or has_val_loss:
                axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 파일로 저장
            plot_path = os.path.join(plots_dir, f"learning_curves_{subset}_{time_str}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 학습 곡선 그래프 저장됨: {plot_path}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ 학습 곡선 그래프 저장 중 오류 발생: {e}")
        return False

def save_performance_comparison_plot(all_results, mode, time_str):
    """성능 비교 그래프를 PNG 파일로 저장"""
    try:
        # 결과 디렉토리 확인
        plots_dir = os.path.join("results", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 성능 데이터 수집
        performance_data = {subset: {} for subset in all_results.keys()}
        
        for subset, seed_results in all_results.items():
            auc_roc_values = []
            auc_pr_values = []
            eer_values = []
            
            for seed, results in seed_results.items():
                if results:
                    if 'auc_roc' in results:
                        auc_roc_values.append(results['auc_roc'])
                    if 'auc_pr' in results:
                        auc_pr_values.append(results['auc_pr'])
                    if 'eer' in results:
                        eer_values.append(results['eer'])
            
            if auc_roc_values:
                performance_data[subset]['auc_roc_mean'] = np.mean(auc_roc_values)
                performance_data[subset]['auc_roc_std'] = np.std(auc_roc_values)
            if auc_pr_values:
                performance_data[subset]['auc_pr_mean'] = np.mean(auc_pr_values)
                performance_data[subset]['auc_pr_std'] = np.std(auc_pr_values)
            if eer_values:
                performance_data[subset]['eer_mean'] = np.mean(eer_values)
                performance_data[subset]['eer_std'] = np.std(eer_values)
        
        # 유효한 데이터가 있는 부위만 필터링
        valid_subsets = [subset for subset, data in performance_data.items() 
                        if data and any('mean' in key for key in data.keys())]
        
        if not valid_subsets:
            print("📊 성능 비교 그래프를 생성할 데이터가 없습니다.")
            return False
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Across Joint Subsets', fontsize=16)
        
        x_pos = np.arange(len(valid_subsets))
        width = 0.35
        
        # AUC(ROC) 비교
        auc_roc_means = [performance_data[subset].get('auc_roc_mean', 0) for subset in valid_subsets]
        auc_roc_stds = [performance_data[subset].get('auc_roc_std', 0) for subset in valid_subsets]
        
        if any(auc_roc_means):
            bars1 = axes[0, 0].bar(x_pos, auc_roc_means, width, yerr=auc_roc_stds, 
                                  capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
            axes[0, 0].set_title('AUC(ROC) Comparison')
            axes[0, 0].set_ylabel('AUC(ROC)')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(valid_subsets, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 값 표시
            for i, (mean, std) in enumerate(zip(auc_roc_means, auc_roc_stds)):
                if mean > 0:
                    axes[0, 0].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                                   ha='center', va='bottom', fontsize=8)
        
        # AUC(PR) 비교
        auc_pr_means = [performance_data[subset].get('auc_pr_mean', 0) for subset in valid_subsets]
        auc_pr_stds = [performance_data[subset].get('auc_pr_std', 0) for subset in valid_subsets]
        
        if any(auc_pr_means):
            bars2 = axes[0, 1].bar(x_pos, auc_pr_means, width, yerr=auc_pr_stds, 
                                  capsize=5, alpha=0.8, color='lightcoral', edgecolor='darkred')
            axes[0, 1].set_title('AUC(PR) Comparison')
            axes[0, 1].set_ylabel('AUC(PR)')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(valid_subsets, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 값 표시
            for i, (mean, std) in enumerate(zip(auc_pr_means, auc_pr_stds)):
                if mean > 0:
                    axes[0, 1].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                                   ha='center', va='bottom', fontsize=8)
        
        # EER 비교 (낮을수록 좋음)
        eer_means = [performance_data[subset].get('eer_mean', 0) for subset in valid_subsets]
        eer_stds = [performance_data[subset].get('eer_std', 0) for subset in valid_subsets]
        
        if any(eer_means):
            bars3 = axes[1, 0].bar(x_pos, eer_means, width, yerr=eer_stds, 
                                  capsize=5, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
            axes[1, 0].set_title('EER Comparison (Lower is Better)')
            axes[1, 0].set_ylabel('EER')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(valid_subsets, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 값 표시
            for i, (mean, std) in enumerate(zip(eer_means, eer_stds)):
                if mean > 0:
                    axes[1, 0].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                                   ha='center', va='bottom', fontsize=8)
        
        # 전체 성능 종합 비교 (radar chart 스타일의 선 그래프)
        if any(auc_roc_means) and any(auc_pr_means):
            ax_combined = axes[1, 1]
            x_labels = [subset[:8] for subset in valid_subsets]  # 라벨 축약
            
            # 정규화 (0-1 스케일)
            if max(auc_roc_means) > 0:
                norm_auc_roc = [x/max(auc_roc_means) for x in auc_roc_means]
                ax_combined.plot(x_labels, norm_auc_roc, 'o-', label='AUC(ROC)', linewidth=2, markersize=6)
            
            if max(auc_pr_means) > 0:
                norm_auc_pr = [x/max(auc_pr_means) for x in auc_pr_means]
                ax_combined.plot(x_labels, norm_auc_pr, 's-', label='AUC(PR)', linewidth=2, markersize=6)
            
            if max(eer_means) > 0:
                # EER은 낮을수록 좋으므로 반전
                norm_eer = [(max(eer_means) - x)/max(eer_means) for x in eer_means]
                ax_combined.plot(x_labels, norm_eer, '^-', label='1-EER (normalized)', linewidth=2, markersize=6)
            
            ax_combined.set_title('Normalized Performance Comparison')
            ax_combined.set_ylabel('Normalized Score')
            ax_combined.set_ylim(0, 1.1)
            ax_combined.legend()
            ax_combined.grid(True, alpha=0.3)
            plt.setp(ax_combined.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 파일로 저장
        plot_path = os.path.join(plots_dir, f"performance_comparison_{time_str}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 성능 비교 그래프 저장됨: {plot_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ 성능 비교 그래프 저장 중 오류 발생: {e}")
        return False

def save_learning_rate_convergence_plot(all_results, time_str):
    """Learning Rate 수렴 패턴을 보여주는 전용 그래프를 PNG 파일로 저장"""
    try:
        # 결과 디렉토리 확인
        plots_dir = os.path.join("results", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Learning rate 데이터 수집
        lr_data_collection = {}
        
        for subset, seed_results in all_results.items():
            if not seed_results:
                continue
                
            lr_data_collection[subset] = []
            for seed, results in seed_results.items():
                if results and 'learning_curves' in results:
                    learning_data = results['learning_curves']
                    if learning_data['learning_rates']:
                        lr_data_collection[subset].append({
                            'seed': seed,
                            'learning_rates': learning_data['learning_rates'],
                            'epochs': list(range(1, len(learning_data['learning_rates']) + 1))
                        })
        
        # 유효한 데이터가 있는지 확인
        valid_subsets = [subset for subset, data in lr_data_collection.items() if data]
        
        if not valid_subsets:
            print("📊 Learning Rate 수렴 그래프를 생성할 데이터가 없습니다.")
            return False
        
        # 각 부위별로 별도 그래프 생성 (간단한 방식)
        for subset in valid_subsets:
            lr_data_list = lr_data_collection[subset]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle(f'Learning Rate Convergence - {subset.upper()}', fontsize=14)
            
            # 색상 팔레트
            colors = plt.cm.Set1(np.linspace(0, 1, len(lr_data_list)))
            
            for i, lr_info in enumerate(lr_data_list):
                seed = lr_info['seed']
                epochs = lr_info['epochs']
                learning_rates = lr_info['learning_rates']
                color = colors[i]
                
                # 시드별로 다른 스타일 적용
                linestyles = ['-', '--', '-.', ':', '-']
                markers = ['o', 's', '^', 'D', 'v']
                linestyle = linestyles[i % len(linestyles)]
                marker = markers[i % len(markers)]
                
                # Learning rate 곡선 그리기 (스타일 차별화)
                ax.plot(epochs, learning_rates, linestyle=linestyle, color=color, 
                       label=f'Seed {seed}', linewidth=2.5, marker=marker, 
                       markersize=6, alpha=0.8)
                
                # 수렴 값 표시 (마지막 값)
                final_lr = learning_rates[-1]
                ax.annotate(f'{final_lr:.2e}', 
                           xy=(epochs[-1], final_lr), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=9, color=color, alpha=0.8)
            
            ax.set_title(f'{subset.upper()} - Learning Rate Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 파일로 저장
            plot_path = os.path.join(plots_dir, f"learning_rate_{subset}_{time_str}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Learning Rate 그래프 저장됨: {plot_path}")
        
        # 모든 부위 통합 비교 그래프
        if len(valid_subsets) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle('Learning Rate Convergence - All Joint Subsets Comparison', fontsize=14)
            
            subset_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_subsets)))
            
            for subset_idx, subset in enumerate(valid_subsets):
                lr_data_list = lr_data_collection[subset]
                base_color = subset_colors[subset_idx]
                
                # 각 시드별로 약간 다른 색조 사용
                for seed_idx, lr_info in enumerate(lr_data_list):
                    seed = lr_info['seed']
                    epochs = lr_info['epochs']
                    learning_rates = lr_info['learning_rates']
                    
                    # 시드별 스타일 차별화
                    linestyles = ['-', '--', '-.', ':', '-']
                    markers = ['o', 's', '^', 'D', 'v']
                    linestyle = linestyles[seed_idx % len(linestyles)]
                    marker = markers[seed_idx % len(markers)]
                    alpha = 0.7 + 0.3 * (seed_idx / max(1, len(lr_data_list) - 1))
                    
                    label = f'{subset} (seed {seed})'
                    ax.plot(epochs, learning_rates, linestyle=linestyle, color=base_color, 
                           alpha=alpha, label=label, linewidth=2, marker=marker, markersize=4)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 통합 그래프 저장
            combined_plot_path = os.path.join(plots_dir, f"learning_rate_comparison_all_{time_str}.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Learning Rate 통합 비교 그래프 저장됨: {combined_plot_path}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Learning Rate 수렴 그래프 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_results_to_excel(all_results, mode, time_str):
    """결과를 Excel 파일로 저장"""
    try:
        # 결과 디렉토리 확인
        results_dir = os.path.join("results", "excel_files")
        os.makedirs(results_dir, exist_ok=True)
        
        excel_path = os.path.join(results_dir, f"experiment_results_{time_str}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 메인 결과 시트
            if mode == "1" or mode == "2":  # 여러 시드에 대한 결과
                rows = []
                for subset, seed_results in all_results.items():
                    for seed, results in seed_results.items():
                        if results:
                            row = {
                                'Joint_Subset': subset,
                                'Seed': seed,
                                'AUC_ROC': results.get('auc_roc', None),
                                'AUC_PR': results.get('auc_pr', None),
                                'EER': results.get('eer', None),
                                'EER_Threshold': results.get('eer_threshold', None),
                                'Num_Samples': results.get('num_samples', None)
                            }
                            rows.append(row)
                
                if rows:
                    df_main = pd.DataFrame(rows)
                    df_main.to_excel(writer, sheet_name='Main_Results', index=False)
                    
                    # 통계 요약 시트
                    summary_rows = []
                    for subset in all_results.keys():
                        subset_data = df_main[df_main['Joint_Subset'] == subset]
                        if not subset_data.empty:
                            summary_row = {
                                'Joint_Subset': subset,
                                'Num_Seeds': len(subset_data),
                                'AUC_ROC_Mean': subset_data['AUC_ROC'].mean() if 'AUC_ROC' in subset_data.columns else None,
                                'AUC_ROC_Std': subset_data['AUC_ROC'].std() if 'AUC_ROC' in subset_data.columns else None,
                                'AUC_PR_Mean': subset_data['AUC_PR'].mean() if 'AUC_PR' in subset_data.columns else None,
                                'AUC_PR_Std': subset_data['AUC_PR'].std() if 'AUC_PR' in subset_data.columns else None,
                                'EER_Mean': subset_data['EER'].mean() if 'EER' in subset_data.columns else None,
                                'EER_Std': subset_data['EER'].std() if 'EER' in subset_data.columns else None,
                            }
                            summary_rows.append(summary_row)
                    
                    if summary_rows:
                        df_summary = pd.DataFrame(summary_rows)
                        df_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            elif mode == "3":  # 단일 시드, 여러 관절 부위에 대한 결과
                seed = list(list(all_results.values())[0].keys())[0]
                rows = []
                for subset, seed_results in all_results.items():
                    if seed in seed_results and seed_results[seed]:
                        results = seed_results[seed]
                        row = {
                            'Joint_Subset': subset,
                            'Seed': seed,
                            'AUC_ROC': results.get('auc_roc', None),
                            'AUC_PR': results.get('auc_pr', None),
                            'EER': results.get('eer', None),
                            'EER_Threshold': results.get('eer_threshold', None),
                            'Num_Samples': results.get('num_samples', None)
                        }
                        rows.append(row)
                
                if rows:
                    df_main = pd.DataFrame(rows)
                    df_main.to_excel(writer, sheet_name='Results', index=False)
            
            # 학습 곡선 데이터 시트 (있는 경우)
            learning_curve_rows = []
            for subset, seed_results in all_results.items():
                for seed, results in seed_results.items():
                    if results and 'learning_curves' in results:
                        learning_data = results['learning_curves']
                        max_len = max(
                            len(learning_data.get('epochs', [])),
                            len(learning_data.get('train_loss', [])),
                            len(learning_data.get('val_loss', [])),
                            len(learning_data.get('learning_rates', []))
                        )
                        
                        for i in range(max_len):
                            row = {
                                'Joint_Subset': subset,
                                'Seed': seed,
                                'Step': i + 1,
                                'Epoch': learning_data['epochs'][i] if i < len(learning_data['epochs']) else None,
                                'Train_Loss': learning_data['train_loss'][i] if i < len(learning_data['train_loss']) else None,
                                'Val_Loss': learning_data['val_loss'][i] if i < len(learning_data['val_loss']) else None,
                                'Learning_Rate': learning_data['learning_rates'][i] if i < len(learning_data['learning_rates']) else None
                            }
                            learning_curve_rows.append(row)
            
            if learning_curve_rows:
                df_learning = pd.DataFrame(learning_curve_rows)
                df_learning.to_excel(writer, sheet_name='Learning_Curves', index=False)
            
            # 실험 정보 시트
            info_data = {
                'Parameter': ['Experiment_Date', 'Mode', 'Total_Experiments', 'File_Generated'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"Mode {mode}",
                    sum(len(seed_results) for seed_results in all_results.values()),
                    time_str
                ]
            }
            df_info = pd.DataFrame(info_data)
            df_info.to_excel(writer, sheet_name='Experiment_Info', index=False)
        
        print(f"📊 결과가 Excel 파일로 저장되었습니다: {excel_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ Excel 파일 저장 중 오류 발생: {e}")
        return False

def main():
    # 실행할 관절 부위 목록
    subsets = ["all", "arms", "legs", "body", "head", "left_arm", "right_arm", "left_leg", "right_leg","arm+body", "head+body"]
    
    # 실행할 시드 목록 (여기에 원하는 시드 값을 추가하세요)
    seeds = [42,998,75,38,142]
    
    # 사용자에게 어떤 모드로 실행할지 묻기
    print("실험 실행 모드를 선택하세요:")
    print("1. 모든 관절 부위 x 모든 시드 (총 " + str(len(subsets) * len(seeds)) + "개 실험)")
    print("2. 특정 관절 부위에 대해 여러 시드로 실험")
    print("3. 여러 관절 부위에 대해 단일 시드로 실험")
    
    mode = input("선택 (1/2/3): ").strip()
    
    # 결과 저장 딕셔너리 초기화
    all_results = {}
    
    start_time = time.time()
    
    if mode == "1":
        # 모든 조합 실행
        print(f"🔬 모든 관절 부위와 시드 조합으로 실험 시작 (총 {len(subsets) * len(seeds)}개 실험)")
        print("=" * 60)
        
        for subset in subsets:
            all_results[subset] = {}
            
            for seed in seeds:
                experiment_key = f"{subset}_seed{seed}"
                success, results = run_experiment(subset, seed)
                
                if success and results:
                    all_results[subset][seed] = results
                else:
                    all_results[subset][seed] = None
    
    elif mode == "2":
        # 특정 관절 부위에 대해 여러 시드로 실험
        print("사용할 관절 부위를 선택하세요:")
        for i, subset in enumerate(subsets, 1):
            print(f"{i}. {subset}")
        
        subset_idx = int(input("번호 선택: ").strip()) - 1
        if 0 <= subset_idx < len(subsets):
            subset = subsets[subset_idx]
            all_results[subset] = {}
            
            print(f"🔬 '{subset}' 관절 부위에 대해 {len(seeds)}개 시드로 실험 시작")
            print("=" * 60)
            
            for seed in seeds:
                success, results = run_experiment(subset, seed)
                
                if success and results:
                    all_results[subset][seed] = results
                else:
                    all_results[subset][seed] = None
        else:
            print("❌ 잘못된 선택입니다.")
            return
            
    elif mode == "3":
        # 여러 관절 부위에 대해 단일 시드로 실험
        print("사용할 시드를 선택하세요:")
        for i, seed in enumerate(seeds, 1):
            print(f"{i}. {seed}")
        
        seed_idx = int(input("번호 선택: ").strip()) - 1
        if 0 <= seed_idx < len(seeds):
            seed = seeds[seed_idx]
            
            print(f"🔬 {len(subsets)}개 관절 부위에 대해 시드 {seed}로 실험 시작")
            print("=" * 60)
            
            for subset in subsets:
                success, results = run_experiment(subset, seed)
                
                if success and results:
                    all_results[subset] = {seed: results}
                else:
                    all_results[subset] = {seed: None}
        else:
            print("❌ 잘못된 선택입니다.")
            return
            
    else:
        print("❌ 잘못된 선택입니다.")
        return
    
    # 결과 테이블 출력
    total_time = time.time() - start_time
    print(f"\n{'='*100}")
    print("📊 EXPERIMENT RESULTS")
    print(f"{'='*100}")
    
    # 모드에 따라 다르게 결과 출력
    if mode == "1" or mode == "2":  # 여러 시드에 대한 결과
        # 각 관절 부위별로 결과 출력
        for subset, seed_results in all_results.items():
            print(f"\n{'='*50}")
            print(f"🔍 {subset.upper()} 부위 결과")
            print(f"{'='*50}")
            
            # 헤더 출력
            header = f"{'Seed':<8} {'AUC(ROC)':<10} {'AUC(PR)':<10} {'EER':<8} {'EER_Th':<8} {'Samples':<8}"
            print(header)
            print("-" * len(header))
            
            # 각 시드별 결과 출력
            for seed, results in seed_results.items():
                if results:
                    auc_roc = f"{results.get('auc_roc', 0):.4f}" if 'auc_roc' in results else "N/A"
                    auc_pr = f"{results.get('auc_pr', 0):.4f}" if 'auc_pr' in results else "N/A"
                    eer = f"{results.get('eer', 0):.4f}" if 'eer' in results else "N/A"
                    eer_th = f"{results.get('eer_threshold', 0):.4f}" if 'eer_threshold' in results else "N/A"
                    samples = str(results.get('num_samples', 'N/A'))
                    
                    row = f"{seed:<8} {auc_roc:<10} {auc_pr:<10} {eer:<8} {eer_th:<8} {samples:<8}"
                else:
                    row = f"{seed:<8} {'FAILED':<10} {'FAILED':<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}"
                
                print(row)
            
            # 시드 평균 계산
            valid_results = [r for r in seed_results.values() if r]
            if valid_results:
                avg_auc_roc = sum(r.get('auc_roc', 0) for r in valid_results) / len(valid_results)
                avg_auc_pr = sum(r.get('auc_pr', 0) for r in valid_results) / len(valid_results)
                avg_eer = sum(r.get('eer', 0) for r in valid_results) / len(valid_results)
                
                print("-" * len(header))
                print(f"{'평균':<8} {avg_auc_roc:.4f}   {avg_auc_pr:.4f}   {avg_eer:.4f}")
            
    elif mode == "3":  # 단일 시드, 여러 관절 부위에 대한 결과
        # 사용한 시드 값 가져오기
        seed = list(list(all_results.values())[0].keys())[0]
        
        # 헤더 출력
        header = f"{'Joint Part':<12} {'AUC(ROC)':<10} {'AUC(PR)':<10} {'EER':<8} {'EER_Th':<8} {'Samples':<8}"
        print(header)
        print("-" * len(header))
        
        # 각 관절 부위별 결과 출력
        for subset, seed_results in all_results.items():
            results = seed_results.get(seed)
            
            if results:
                auc_roc = f"{results.get('auc_roc', 0):.4f}" if 'auc_roc' in results else "N/A"
                auc_pr = f"{results.get('auc_pr', 0):.4f}" if 'auc_pr' in results else "N/A"
                eer = f"{results.get('eer', 0):.4f}" if 'eer' in results else "N/A"
                eer_th = f"{results.get('eer_threshold', 0):.4f}" if 'eer_threshold' in results else "N/A"
                samples = str(results.get('num_samples', 'N/A'))
                
                row = f"{subset:<12} {auc_roc:<10} {auc_pr:<10} {eer:<8} {eer_th:<8} {samples:<8}"
            else:
                row = f"{subset:<12} {'FAILED':<10} {'FAILED':<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}"
            
            print(row)
        
        # 성능 비교 (성공한 실험만)
        successful_results = {}
        for subset, seed_results in all_results.items():
            if seed in seed_results and seed_results[seed]:
                successful_results[subset] = seed_results[seed]
        
        if len(successful_results) > 1:
            print(f"\n{'='*50}")
            print(f"🏆 PERFORMANCE RANKING (Seed: {seed})")
            print(f"{'='*50}")
            
            # AUC(ROC) 기준 정렬
            if all('auc_roc' in results for results in successful_results.values()):
                sorted_by_auc = sorted(successful_results.items(), 
                                     key=lambda x: x[1]['auc_roc'], reverse=True)
                
                print("By AUC(ROC):")
                for i, (subset, results) in enumerate(sorted_by_auc, 1):
                    print(f"{i}. {subset:<12}: {results['auc_roc']:.4f}")
            
            # EER 기준 정렬 (낮을수록 좋음)
            if all('eer' in results for results in successful_results.values()):
                sorted_by_eer = sorted(successful_results.items(), 
                                     key=lambda x: x[1]['eer'])
                
                print(f"\nBy EER (lower is better):")
                for i, (subset, results) in enumerate(sorted_by_eer, 1):
                    print(f"{i}. {subset:<12}: {results['eer']:.4f}")
    
    # Excel 파일로 결과 저장 및 학습 곡선 그래프 생성
    time_str = time.strftime("%Y%m%d_%H%M%S")
    
    # Excel 파일 저장
    excel_success = save_results_to_excel(all_results, mode, time_str)
    
    # 학습 곡선 그래프 생성 및 저장
    plot_success = save_learning_curves_plot(all_results, time_str)
    
    # 성능 비교 그래프 생성 및 저장
    performance_plot_success = save_performance_comparison_plot(all_results, mode, time_str)
    
    # Learning Rate 전용 그래프 생성 및 저장  
    lr_plot_success = save_learning_rate_convergence_plot(all_results, time_str)
    
    # 기존 CSV 파일도 유지 (호환성을 위해)
    try:
        # 결과 디렉토리 확인
        results_dir = os.path.join("results", "csv_files")
        os.makedirs(results_dir, exist_ok=True)
        
        # 결과를 데이터프레임으로 변환
        if mode == "1" or mode == "2":  # 여러 시드에 대한 결과
            rows = []
            for subset, seed_results in all_results.items():
                for seed, results in seed_results.items():
                    if results:
                        row = {
                            'joint_subset': subset,
                            'seed': seed,
                            'auc_roc': results.get('auc_roc', None),
                            'auc_pr': results.get('auc_pr', None),
                            'eer': results.get('eer', None),
                            'eer_threshold': results.get('eer_threshold', None),
                            'num_samples': results.get('num_samples', None)
                        }
                        rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(results_dir, f"experiment_results_{time_str}.csv")
                df.to_csv(csv_path, index=False)
                print(f"📄 결과가 CSV 파일로도 저장되었습니다: {csv_path}")
            
        elif mode == "3":  # 단일 시드, 여러 관절 부위에 대한 결과
            seed = list(list(all_results.values())[0].keys())[0]
            rows = []
            for subset, seed_results in all_results.items():
                if seed in seed_results and seed_results[seed]:
                    results = seed_results[seed]
                    row = {
                        'joint_subset': subset,
                        'seed': seed,
                        'auc_roc': results.get('auc_roc', None),
                        'auc_pr': results.get('auc_pr', None),
                        'eer': results.get('eer', None),
                        'eer_threshold': results.get('eer_threshold', None),
                        'num_samples': results.get('num_samples', None)
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(results_dir, f"experiment_results_seed{seed}_{time_str}.csv")
                df.to_csv(csv_path, index=False)
                print(f"📄 결과가 CSV 파일로도 저장되었습니다: {csv_path}")
    
    except Exception as e:
        print(f"⚠️ CSV 파일 저장 중 오류 발생: {e}")
    
    print(f"\n⏱️ 총 실행 시간: {total_time:.1f}초")
    print("\n🎉 모든 실험이 완료되었습니다!")

if __name__ == "__main__":
    main()