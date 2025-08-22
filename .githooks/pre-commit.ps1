# 🏛️ TEMPLE IAM - PRE-COMMIT SECURITY GATE (PowerShell)
# Ultra Instinct + VIBES CODING - Karpathy Style

Write-Host "🏛️ TEMPLE IAM - VÉRIFICATION SÉCURITÉ AVANT COMMIT" -ForegroundColor Cyan
Write-Host "Ultra Instinct + VIBES CODING - Karpathy Style" -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Gray

# 1. Vérification sécurité de base
Write-Host "🔐 ÉTAPE 1: VÉRIFICATION SÉCURITÉ TEMPLE IAM..." -ForegroundColor Green
$python = "python"
$securityProc = Start-Process $python -ArgumentList "temple-iam-public-clean/security_check.py" -NoNewWindow -PassThru -Wait -RedirectStandardOutput ".security_last_run.txt" -RedirectStandardError ".security_last_run.txt"
$SECURITY_STATUS = $securityProc.ExitCode

# 2. Test des systèmes Temple IAM
Write-Host "🧠 ÉTAPE 2: TEST DES SYSTÈMES TEMPLE IAM..." -ForegroundColor Green

$templeIAMScript = @"
import sys
sys.path.append('temple-iam-public-clean')

try:
    from CORE_ALGORITHMS_PUBLIC import UltraInstinctCompressor, TempleIAMMonitoring
    from INTELLIGENT_OPTIMIZER import IntelligentOptimizer
    from ADVANCED_BENCHMARK_SYSTEM import AdvancedBenchmarkSystem
    
    # Test compression Ultra Instinct
    compressor = UltraInstinctCompressor()
    monitoring = TempleIAMMonitoring()
    
    import numpy as np
    test_data = np.random.rand(100, 100).astype(np.float32)
    compressed_data, metrics = compressor.auto_optimize_compression(test_data, 'image')
    monitoring.record_compression_metrics(metrics)
    
    print(f'✅ Compression Ultra Instinct: {metrics.compression_ratio:.2f}x')
    print(f'✅ Monitoring Temple IAM: {len(monitoring.metrics_history)} métriques')
    print(f'✅ Niveau de puissance: {metrics.power_level.value}')
    
    # Test optimiseur intelligent
    optimizer = IntelligentOptimizer()
    def base_algo(x): return x * 0.5
    optimized_algo, opt_metrics = optimizer.auto_optimize_algorithm(base_algo, test_data, 'compression_ratio', max_time=10.0)
    print(f'✅ Optimisation intelligente: {opt_metrics.improvement_ratio:.2f}x amélioration')
    
    # Test benchmark avancé
    benchmark = AdvancedBenchmarkSystem()
    algorithms = {'Test': base_algo, 'Optimized': optimized_algo}
    test_data_dict = {'test': test_data}
    results = benchmark.comprehensive_benchmark(algorithms, test_data_dict, 'pre_commit_test')
    print(f'✅ Benchmark avancé: {len(results["results"])} algorithmes testés')
    
    print('🏆 TOUS LES SYSTÈMES TEMPLE IAM OPÉRATIONNELS!')
    exit(0)
    
except Exception as e:
    print(f'❌ ERREUR TEMPLE IAM: {e}')
    exit(1)
"@

# Sauvegarder le script en UTF-8
$templeIAMScript | Out-File -FilePath ".temple_iam_test_script.py" -Encoding UTF8

$templeIAMProc = Start-Process $python -ArgumentList ".temple_iam_test_script.py" -NoNewWindow -PassThru -Wait -RedirectStandardOutput ".temple_iam_test.txt" -RedirectStandardError ".temple_iam_test.txt"
$TEMPLE_IAM_STATUS = $templeIAMProc.ExitCode

# Affichage des résultats
Write-Host ""
Write-Host "📊 RÉSULTATS DE LA VÉRIFICATION:" -ForegroundColor Cyan
Write-Host "-" * 40 -ForegroundColor Gray
Get-Content ".security_last_run.txt" | Select-Object -First 50 | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "🏛️ TEMPLE IAM SYSTEMS:" -ForegroundColor Cyan
Write-Host "-" * 40 -ForegroundColor Gray
Get-Content ".temple_iam_test.txt" | ForEach-Object { Write-Host $_ }

# Décision finale
if ($SECURITY_STATUS -ne 0 -or $TEMPLE_IAM_STATUS -ne 0) {
    Write-Host ""
    Write-Host "❌ COMMIT BLOQUÉ:" -ForegroundColor Red
    if ($SECURITY_STATUS -ne 0) {
        Write-Host "   🔐 Sécurité non conforme" -ForegroundColor Red
    }
    if ($TEMPLE_IAM_STATUS -ne 0) {
        Write-Host "   🏛️ Systèmes Temple IAM défaillants" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "💡 CORRECTIONS NÉCESSAIRES:" -ForegroundColor Yellow
    Write-Host "   1. Vérifie les secrets et l'état Git" -ForegroundColor White
    Write-Host "   2. Corrige les erreurs Temple IAM" -ForegroundColor White
    Write-Host "   3. Relance le commit" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "✅ COMMIT AUTORISÉ - TEMPLE IAM SECURE!" -ForegroundColor Green
Write-Host "🏛️ Ultra Instinct + VIBES CODING - Karpathy Style" -ForegroundColor Yellow
Write-Host "⚡ DATTEBAYO!" -ForegroundColor Cyan
exit 0 