#!/usr/bin/env python3
"""
Script de setup e verificação do projeto.
Verifica dependências e configuração.
"""

import sys
import subprocess
import platform
import shutil


def print_header(text):
    """Imprime um cabeçalho formatado."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_python_version():
    """Verifica a versão do Python."""
    print_header("Verificando Python")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        return False
    
    print("✅ Versão do Python adequada")
    return True


def check_pip():
    """Verifica se pip está instalado."""
    print_header("Verificando pip")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout.strip())
        print("✅ pip está instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip não encontrado")
        return False


def check_ollama():
    """Verifica se Ollama está instalado."""
    print_header("Verificando Ollama")
    
    ollama_path = shutil.which("ollama")
    
    if not ollama_path:
        print("❌ Ollama não encontrado")
        print("\nPara instalar:")
        
        os_name = platform.system()
        if os_name == "Darwin":  # macOS
            print("  brew install ollama")
            print("  ou visite: https://ollama.ai/download")
        elif os_name == "Linux":
            print("  curl -fsSL https://ollama.ai/install.sh | sh")
        elif os_name == "Windows":
            print("  Baixe em: https://ollama.ai/download")
        
        return False
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Ollama encontrado: {ollama_path}")
        print(result.stdout.strip())
        print("✅ Ollama está instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao executar Ollama")
        return False


def install_dependencies():
    """Instala as dependências do projeto."""
    print_header("Instalando Dependências")
    
    try:
        print("Instalando pacotes do requirements.txt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Dependências instaladas com sucesso")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False


def check_ollama_running():
    """Verifica se o Ollama está rodando."""
    print_header("Verificando Serviço Ollama")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✅ Ollama está rodando")
            
            data = response.json()
            models = data.get('models', [])
            
            if models:
                print(f"\n📚 Modelos disponíveis ({len(models)}):")
                for model in models[:5]:  # Mostra até 5 modelos
                    print(f"  • {model['name']}")
                if len(models) > 5:
                    print(f"  ... e mais {len(models) - 5} modelos")
            else:
                print("\n⚠️  Nenhum modelo baixado ainda")
                print("Execute o programa para baixar o modelo automaticamente")
            
            return True
        else:
            print("⚠️  Ollama instalado mas não está rodando")
            print("\nPara iniciar:")
            print("  ollama serve")
            return False
            
    except Exception as e:
        print("⚠️  Ollama instalado mas não está rodando")
        print(f"Erro: {e}")
        print("\nPara iniciar:")
        print("  ollama serve")
        return False


def create_venv():
    """Cria um ambiente virtual."""
    print_header("Criando Ambiente Virtual")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "venv"],
            check=True
        )
        print("✅ Ambiente virtual criado em: ./venv")
        
        os_name = platform.system()
        if os_name == "Windows":
            activate_cmd = ".\\venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print(f"\nPara ativar:")
        print(f"  {activate_cmd}")
        
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao criar ambiente virtual")
        return False


def show_next_steps(all_ok):
    """Mostra os próximos passos."""
    print_header("Próximos Passos")
    
    if all_ok:
        print("\n✅ Setup completo! Você pode:")
        print("\n1. Executar o chat interativo:")
        print("   python -m src.main")
        print("\n2. Executar os exemplos:")
        print("   python exemplos.py")
        print("\n3. Executar os testes:")
        print("   python -m tests.test_models")
        print("\n4. Fazer uma consulta rápida:")
        print('   python -m src.main "Sua pergunta aqui"')
    else:
        print("\n⚠️  Algumas verificações falharam.")
        print("Por favor, resolva os problemas acima antes de continuar.")


def main():
    """Função principal."""
    print("\n" + "🤖 " + "="*56 + " 🤖")
    print("  SETUP - Projeto Ollama LLM")
    print("  POO, SOLID e Clean Code")
    print("🤖 " + "="*56 + " 🤖")
    
    checks = []
    
    # Verificações
    checks.append(("Python", check_python_version()))
    checks.append(("pip", check_pip()))
    checks.append(("Ollama", check_ollama()))
    
    # Se tudo OK até agora, instala dependências
    if all(result for _, result in checks):
        checks.append(("Dependências", install_dependencies()))
        checks.append(("Serviço Ollama", check_ollama_running()))
    
    # Resumo
    print_header("Resumo")
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_ok = all(result for _, result in checks)
    show_next_steps(all_ok)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
