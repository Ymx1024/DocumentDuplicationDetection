#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask应用主文件 - 整合所有蓝图和功能模块
"""

from flask import Flask, render_template

# 创建Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 导入蓝图
from single_compare import single_compare_bp
from storage_compare import storage_compare_bp
from keyword_search import keyword_search_bp

# 注册蓝图
print("🔧 开始注册蓝图...")
app.register_blueprint(single_compare_bp, url_prefix='/single_compare')
print("✅ 已注册蓝图: single_compare (URL前缀: /single_compare)")

app.register_blueprint(storage_compare_bp, url_prefix='/storage_compare')
print("✅ 已注册蓝图: storage_compare (URL前缀: /storage_compare)")

app.register_blueprint(keyword_search_bp, url_prefix='/keyword_search')
print("✅ 已注册蓝图: keyword_search (URL前缀: /keyword_search)")

print("🔧 蓝图注册完成！")

# 统计注册的蓝图
blueprints = [bp.name for bp in app.blueprints.values()]
print(f"📋 总共注册了 {len(blueprints)} 个蓝图: {blueprints}")

# 显示所有路由
print(f"\n🌐 所有注册的路由:")
for rule in app.url_map.iter_rules():
    methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
    print(f"  {rule.rule} [{methods}] -> {rule.endpoint}")

print("=" * 50)

@app.route('/')
def index():
    """主页路由"""
    from flask import request
    from utils import get_user_id_from_request, register_user_session, cancel_user_tasks, update_system_resources
    
    # 获取用户ID
    user_id = get_user_id_from_request(request)
    
    # 注册用户会话
    register_user_session(user_id, request)
    
    # 检测页面刷新，只取消当前用户的任务
    cancelled_count = cancel_user_tasks(user_id)
    if cancelled_count > 0:
        print(f"用户 {user_id} 主页访问：检测到页面刷新，已取消 {cancelled_count} 个任务")
    
    # 更新系统资源统计
    system_resources = update_system_resources()
    print(f"系统资源状态：活跃用户 {system_resources['active_users']} 个，活跃任务 {system_resources['active_tasks']} 个")
    
    return render_template('index.html')

@app.route('/api/page_visibility', methods=['POST'])
def handle_page_visibility():
    """处理页面可见性变化"""
    from flask import request, jsonify
    from utils import get_user_id_from_request, mark_user_page_hidden, mark_user_page_visible
    
    try:
        data = request.get_json()
        visibility = data.get('visibility', 'visible')
        user_id = get_user_id_from_request(request)
        
        if visibility == 'hidden':
            mark_user_page_hidden(user_id)
            print(f"用户 {user_id} 页面已隐藏")
        else:
            mark_user_page_visible(user_id)
            print(f"用户 {user_id} 页面已可见")
        
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        print(f"处理页面可见性变化时出错: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # 启动用户清理检查器
    from utils import start_user_cleanup_checker
    start_user_cleanup_checker()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
