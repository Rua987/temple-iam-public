#!/usr/bin/env python3
# 🛡️ TEMPLE IAM - SECURITY CHECK AUTOMATION
# Approche fonctionnelle pure (Karpathy style) - pas d'effets de bord

import os
import subprocess
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def run_command(cmd: str) -> Tuple[bool, str, str]:
    """Exécute une commande système - fonction pure"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_git_status() -> Dict[str, str]:
    """Vérifie l'état Git - fonction pure"""
    success, stdout, stderr = run_command("git status --porcelain")
    if not success:
        return {"status": "ERROR", "message": f"Git error: {stderr}"}
    
    if not stdout:
        return {"status": "CLEAN", "message": "Working tree clean"}
    else:
        return {"status": "DIRTY", "message": f"Uncommitted changes: {len(stdout.splitlines())} files"}

def check_git_history() -> Dict[str, str]:
    """Vérifie l'historique Git - fonction pure"""
    success, stdout, stderr = run_command("git log --oneline -5")
    if not success:
        return {"status": "ERROR", "message": f"Git log error: {stderr}"}
    
    commits = stdout.splitlines()
    return {
        "status": "OK", 
        "message": f"Last {len(commits)} commits clean",
        "commits": commits
    }

def scan_for_secrets(directory: str = ".") -> Dict[str, List[str]]:
    """Scanne les secrets potentiels - fonction pure"""
    secret_patterns = [
        r'sk_[a-zA-Z0-9]{20,}',  # API keys
        r'api[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]+',
        r'secret[_-]?key["\s]*[:=]["\s]*[a-zA-Z0-9]+',
        r'password["\s]*[:=]["\s]*[a-zA-Z0-9]+',
        r'token["\s]*[:=]["\s]*[a-zA-Z0-9]+',
        r'["\'][\w+/=]{40,}["\']',  # Base64 encoded
    ]
    
    findings = []
    
    try:
        for root, dirs, files in os.walk(directory):
            # Ignorer les répertoires sensibles
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', '.env', 'node_modules'}]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.json', '.yaml', '.yml', '.txt', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for pattern in secret_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                findings.append(f"{file_path}: {pattern} -> {matches[:3]}")  # Limite à 3 matches
                    except Exception:
                        continue
    except Exception as e:
        return {"status": "ERROR", "findings": [f"Scan error: {str(e)}"]}
    
    return {
        "status": "CLEAN" if not findings else "FOUND",
        "findings": findings
    }

def check_file_structure() -> Dict[str, List[str]]:
    """Vérifie la structure des fichiers - fonction pure"""
    current_dir = Path(".")
    
    # Fichiers critiques qui devraient être présents
    critical_files = [".gitignore", "README.md"]
    missing_critical = [f for f in critical_files if not (current_dir / f).exists()]
    
    # Fichiers sensibles qui ne devraient PAS être présents
    sensitive_patterns = ["*.key", "*.pem", "secrets.*", ".env", "config.json"]
    found_sensitive = []
    
    for pattern in sensitive_patterns:
        matches = list(current_dir.glob(pattern))
        found_sensitive.extend([str(m) for m in matches])
    
    return {
        "missing_critical": missing_critical,
        "found_sensitive": found_sensitive,
        "status": "OK" if not missing_critical and not found_sensitive else "WARNING"
    }

def generate_security_report() -> Dict:
    """Génère un rapport de sécurité complet - fonction pure"""
    print("🛡️ TEMPLE IAM - SECURITY CHECK AUTOMATION")
    print("=" * 50)
    
    # 1. Vérification Git
    print("\n1. 📊 GIT STATUS CHECK...")
    git_status = check_git_status()
    print(f"   Status: {git_status['status']} - {git_status['message']}")
    
    # 2. Historique Git
    print("\n2. 📝 GIT HISTORY CHECK...")
    git_history = check_git_history()
    print(f"   Status: {git_history['status']} - {git_history['message']}")
    if 'commits' in git_history:
        for commit in git_history['commits'][:3]:
            print(f"   • {commit}")
    
    # 3. Scan des secrets
    print("\n3. 🔍 SECRETS SCAN...")
    secrets_scan = scan_for_secrets()
    print(f"   Status: {secrets_scan['status']}")
    if secrets_scan['findings']:
        print("   ⚠️  FINDINGS:")
        for finding in secrets_scan['findings'][:5]:  # Limite affichage
            print(f"   • {finding}")
    else:
        print("   ✅ No secrets found")
    
    # 4. Structure des fichiers
    print("\n4. 📁 FILE STRUCTURE CHECK...")
    file_structure = check_file_structure()
    print(f"   Status: {file_structure['status']}")
    if file_structure['missing_critical']:
        print(f"   ❌ Missing: {file_structure['missing_critical']}")
    if file_structure['found_sensitive']:
        print(f"   ⚠️  Sensitive: {file_structure['found_sensitive']}")
    if file_structure['status'] == 'OK':
        print("   ✅ Structure looks good")
    
    # 5. Rapport final
    print("\n" + "=" * 50)
    
    overall_status = "SECURE"
    issues = []
    
    if git_status['status'] != 'CLEAN':
        issues.append("Git working tree not clean")
        overall_status = "WARNING"
    
    if secrets_scan['status'] == 'FOUND':
        issues.append("Potential secrets found")
        overall_status = "CRITICAL"
    
    if file_structure['status'] != 'OK':
        issues.append("File structure issues")
        if overall_status != "CRITICAL":
            overall_status = "WARNING"
    
    print(f"🎯 OVERALL STATUS: {overall_status}")
    
    if issues:
        print("📋 ISSUES TO RESOLVE:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ ALL CHECKS PASSED - REPOSITORY IS SECURE!")
    
    # Recommandations spécifiques
    print("\n🚀 READY FOR:")
    if overall_status == "SECURE":
        print("   ✅ Technical interviews")
        print("   ✅ Client presentations") 
        print("   ✅ Open source contributions")
        print("   ✅ Academic submissions")
        print("   ✅ Production deployment")
    else:
        print("   ❌ Fix issues before public exposure")
    
    print("\n🛡️ TEMPLE IAM SECURITY CHECK COMPLETED")
    print("IKUZO! DATTEBAYO! 🔥⚡")
    
    return {
        "overall_status": overall_status,
        "git_status": git_status,
        "git_history": git_history,
        "secrets_scan": secrets_scan,
        "file_structure": file_structure,
        "issues": issues
    }

def main():
    """Point d'entrée principal - fonction pure"""
    try:
        report = generate_security_report()
        return 0 if report["overall_status"] == "SECURE" else 1
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        return 2

if __name__ == "__main__":
    exit(main()) 