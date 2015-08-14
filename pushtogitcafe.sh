cd _deploy
# 添加 gitcafe 源
git remote add gitcafe https://gitcafe.com/5long/5long.git >> /dev/null 2>&1
# 提交博客内容
echo "### Pushing to GitCafe..."
git push -u gitcafe master:gitcafe-pages
echo "### Done"%