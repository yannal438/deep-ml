SELECT employees.name, departments.name AS department
FROM employees
INNER JOIN departments 
ON employees.department_id = departments.id;
