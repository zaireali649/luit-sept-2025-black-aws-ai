# Pull Request Template

## 📋 Description

<!-- Provide a brief description of the changes in this PR -->

### What does this PR do?
- [ ] 🆕 New feature
- [ ] 🐛 Bug fix
- [ ] 📚 Documentation update
- [ ] 🎨 Code style/formatting
- [ ] ♻️ Refactoring
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test updates
- [ ] 🔧 Configuration changes
- [ ] 📦 Dependency updates
- [ ] 🤖 ML model updates
- [ ] 📊 Data pipeline changes

### Summary
<!-- Describe what you changed and why -->

### Related Issues
<!-- Link to any related issues using "Fixes #123" or "Closes #123" -->
- Fixes #
- Related to #

---

## 🧪 Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Notebook outputs verified
- [ ] Model performance validated

### Test Results
<!-- Describe the testing performed -->

**Test Environment:**
- Python version: 
- Key dependencies: 
- OS: 

**Test Commands Run:**
```bash
# Example:
pytest tests/ -v
black --check .
flake8 .
jupyter nbconvert --execute notebooks/*.ipynb
```

---

## 🤖 ML/AI Specific Checks

### Model Changes
- [ ] Model architecture changes documented
- [ ] Performance metrics compared (before/after)
- [ ] Model size/complexity impact assessed
- [ ] Training time impact documented
- [ ] Inference time impact measured

### Data Changes
- [ ] Data schema changes documented
- [ ] Data quality checks passed
- [ ] Sample data provided for testing
- [ ] Data privacy/security reviewed
- [ ] Backward compatibility maintained

### Notebook Changes
- [ ] Notebook outputs cleared before commit
- [ ] All cells execute successfully
- [ ] Visualizations render correctly
- [ ] Code follows notebook best practices
- [ ] Documentation cells updated

---

## 📊 Performance Impact

### Metrics
<!-- Include relevant performance metrics -->

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Model Accuracy | | | |
| Training Time | | | |
| Inference Time | | | |
| Memory Usage | | | |
| Model Size | | | |

### Benchmarks
<!-- Include any benchmark results -->

---

## 🔒 Security & Compliance

### Security Checklist
- [ ] No hardcoded credentials or secrets
- [ ] Sensitive data properly handled
- [ ] AWS credentials not exposed
- [ ] API keys properly managed
- [ ] Data privacy requirements met

### AWS Specific
- [ ] IAM permissions documented
- [ ] Resource usage estimated
- [ ] Cost impact assessed
- [ ] Security groups reviewed
- [ ] Data encryption maintained

---

## 📝 Code Quality

### Code Review Checklist
- [ ] Code follows project style guidelines
- [ ] Functions have proper docstrings
- [ ] Error handling implemented
- [ ] Logging appropriately used
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Variable names are descriptive
- [ ] Comments explain complex logic

### DevOps Checklist
- [ ] Dependencies updated in requirements.txt
- [ ] Environment variables documented
- [ ] Configuration changes noted
- [ ] Docker build successful (if applicable)
- [ ] CI/CD pipeline passes
- [ ] Documentation updated

---

## 🚀 Deployment Considerations

### Infrastructure Impact
- [ ] No infrastructure changes required
- [ ] Infrastructure changes documented
- [ ] Resource requirements assessed
- [ ] Scaling implications considered
- [ ] Rollback plan documented

### Deployment Checklist
- [ ] Environment variables configured
- [ ] Database migrations (if applicable)
- [ ] Feature flags configured
- [ ] Monitoring/alerting updated
- [ ] Documentation deployed

---

## 📚 Documentation

### Documentation Updates
- [ ] README.md updated
- [ ] API documentation updated
- [ ] Code comments added/updated
- [ ] Architecture diagrams updated
- [ ] Deployment guides updated
- [ ] User guides updated

### Learning Materials
- [ ] Notebook explanations clear
- [ ] Learning objectives met
- [ ] Prerequisites documented
- [ ] Examples provided
- [ ] Common issues addressed

---

## 🎯 Reviewer Focus Areas

<!-- Highlight specific areas where you want reviewer attention -->

### Please pay special attention to:
- [ ] Algorithm implementation correctness
- [ ] Data processing logic
- [ ] Error handling
- [ ] Performance implications
- [ ] Security considerations
- [ ] Documentation clarity

### Questions for Reviewers:
<!-- List any specific questions you have for reviewers -->

1. 
2. 
3. 

---

## 📸 Screenshots/Visuals

<!-- Include screenshots of new features, updated UIs, or visualization outputs -->

### Before
<!-- Screenshots or outputs before changes -->

### After
<!-- Screenshots or outputs after changes -->

---

## ✅ Pre-merge Checklist

### Author Checklist
- [ ] Self-review completed
- [ ] All tests pass locally
- [ ] Code formatted with Black
- [ ] Linting passes (flake8)
- [ ] Notebook outputs cleared
- [ ] Documentation updated
- [ ] Breaking changes noted
- [ ] Migration guide provided (if needed)

### Reviewer Checklist
- [ ] Code review completed
- [ ] Tests reviewed and adequate
- [ ] Documentation reviewed
- [ ] Security implications considered
- [ ] Performance impact acceptable
- [ ] ML/AI best practices followed

---

## 🔄 Post-merge Tasks

<!-- List any tasks that need to be done after merging -->

- [ ] Deploy to staging environment
- [ ] Update production configuration
- [ ] Monitor performance metrics
- [ ] Update team documentation
- [ ] Notify stakeholders
- [ ] Schedule model retraining (if applicable)

---

## 📞 Additional Context

<!-- Add any additional context, links, or information that reviewers should know -->

### References
- Documentation: 
- Related PRs: 
- External resources: 

### Notes
<!-- Any additional notes for reviewers -->

---

**By submitting this PR, I confirm that:**
- [ ] I have tested these changes thoroughly
- [ ] I have followed the project's contributing guidelines
- [ ] I have considered the impact on existing functionality
- [ ] I am ready to address reviewer feedback promptly
