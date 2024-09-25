/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2022 by the deal.II authors
 *
 * This file is a modified and combined version of the step-12 and step-26 tutorial programs
 * from the deal.II library. The goal is to solve the classical -Fokker-Planck equation
 * as found in 
 * 
 * "Decoherence, Chaos, and the Correspondence Principle",
 *  Habib, Salman & Shizume, Kosuke & Zurek, Wojciech Hubert.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 
 *
 * Author: Guido Kanschat, Texas A&M University, 2009
 *         Wolfgang Bangerth, Texas A&M University, 2013
 *         Timo Heister, Clemson University, 2019
 *         Philipp Baasch, 2024
 */
 
 
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
 
#include <deal.II/meshworker/mesh_loop.h>
 
#include <iostream>
#include <fstream>
 
 
namespace PlanckFokker
{
  using namespace dealii;
 
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
  };
 
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double> &          values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    AssertDimension(values.size(), points.size());
 
    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (points[i](0) < 0.5)
          values[i] = 0.;
        else
          values[i] = 0.;
      }
  }
 
 
 
  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &       mapping,
                const FiniteElement<dim> & fe,
                const Quadrature<dim> &    quadrature,
                const Quadrature<dim - 1> &quadrature_face,
                const UpdateFlags          update_flags = update_values |
                                                 update_gradients |
                                                 update_hessians |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_hessians | update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, quadrature, update_flags)
      , fe_interface_values(mapping,
                            fe,
                            quadrature_face,
                            interface_update_flags)
    {}
 
 
    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}
 
    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };
 
 
 
  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };
 
 
 
  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;
 
    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
 
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };
 
 
  template <int dim>
  class DoubleWell
  {
  public:
    DoubleWell();
    void run();
 
  private:
    void setup_system();
    void assemble_system();
    void solve();
    double compute_norm();
    void refine_grid();
    void output_results() const;
 
    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;
 
    const FE_DGQ<dim> fe;
    DoFHandler<dim>   dof_handler;
 
    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> quadrature_face;
 
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> system_matrix;
 
    Vector<double> old_solution;
    Vector<double> solution;
    Vector<double> right_hand_side;


    double       time;
    double       time_step;
    unsigned int timestep_number;
 
    Tensor<1, dim> beta(const Point<dim> &p);

    const double theta;
    const double D;
  };
 
 
 
  template <int dim>
  DoubleWell<dim>::DoubleWell()
    : mapping()
    , fe(2)
    , dof_handler(triangulation)
    , quadrature(fe.tensor_degree() + 1)
    , quadrature_face(fe.tensor_degree() + 1)
    , time_step(1. / 1000)
    , theta(1.0)
    , D(0.025)
  {}
 
  template <int dim>
  Tensor<1, dim> DoubleWell<dim>::beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());
 
    Tensor<1, dim> hamiltonian_field;
    double A = 10.0;
    double B = 0.5;
    double Omega = 6.07;
    double Gamma = 10;
    hamiltonian_field[0] = p[1];
    hamiltonian_field[1] = -(4*B*pow(p[0], 3) - 2*A*p[0] + Gamma*cos(Omega * time));

    return hamiltonian_field;
  }

  template <int dim>
  class initial_condition : public Function<dim>
  {
    public:
        initial_condition() : Function<dim>(){}
        virtual double value(const Point<dim> & p,
                             const unsigned int component = 0) const override;
  };

  template <int dim>
  double initial_condition<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());
    
    return exp(-(pow(p[0],2) + pow(p[1],2))/2)/(2*M_PI);
  }
 
  template <int dim>
  void DoubleWell<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
 
 
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
 
    mass_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());

    MatrixCreator::create_mass_matrix(  dof_handler,
                                        QGauss<dim>(fe.get_degree() + 1),
                                        mass_matrix);      
    
  }
 
 
 
  template <int dim>
  void DoubleWell<dim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const BoundaryValues<dim> boundary_function;
 
    const auto cell_worker = [&](const Iterator &  cell,
                                 ScratchData<dim> &scratch_data,
                                 CopyData &        copy_data) {
      const unsigned int n_dofs =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);
 
      const auto &q_points = scratch_data.fe_values.get_quadrature_points();
 
      const FEValues<dim> &      fe_v = scratch_data.fe_values;
      const std::vector<double> &JxW  = fe_v.get_JxW_values();
 
      for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
        {
          auto beta_q = beta(q_points[point]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                copy_data.cell_matrix(i, j) +=
                  (time_step * theta * D * 
                   fe_v.shape_grad(j, point)[1] * // grad phi_i[p](x_q)
                   fe_v.shape_grad(i, point)[1] * // grad phi_j[p](x_q)
                   JxW[point]);                   // dx

                copy_data.cell_matrix(i, j) +=
                  (-time_step * theta * 
                   fe_v.shape_grad(i, point)  * beta_q * // grad phi_i(x_q)
                   fe_v.shape_value(j, point) *          // phi_j(x_q)
                   JxW[point]);                          // dx

                copy_data.cell_matrix(i, j) +=
                  (fe_v.shape_value(i, point) * // phi_i(x_q)
                   fe_v.shape_value(j, point) * // phi_j(x_q)
                   JxW[point]);                 // dx
              }
        }
    };
 
    const auto boundary_worker = [&](const Iterator &    cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim> &  scratch_data,
                                     CopyData &          copy_data) {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
        scratch_data.fe_interface_values.get_fe_face_values(0);
 
      const auto &q_points = fe_face.get_quadrature_points();
 
      const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();
      const std::vector<double> &        JxW     = fe_face.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();
 
      std::vector<double> g(q_points.size());
      boundary_function.value_list(q_points, g);
 
      for (unsigned int point = 0; point < q_points.size(); ++point)
        {
          const double beta_dot_n = beta(q_points[point]) * normals[point];

          if (beta_dot_n > 0)
            for (unsigned int i = 0; i < n_facet_dofs; ++i)
              for (unsigned int j = 0; j < n_facet_dofs; ++j)
                copy_data.cell_matrix(i, j) +=
                  time_step * theta *
                  fe_face.shape_value(i, point)   // \phi_i
                  * fe_face.shape_value(j, point) // \phi_j
                  * beta_dot_n                    // \beta . n
                  * JxW[point];                   // dx
        }
    };
 
    const auto face_worker = [&](const Iterator &    cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator &    ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim> &  scratch_data,
                                 CopyData &          copy_data) {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_iv.get_quadrature_points();
 
      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();
 
      const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
 
      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
 
      const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();
 
      for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
        {
          const double beta_dot_n = beta(q_points[qpoint]) * normals[qpoint];
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j){
              copy_data_face.cell_matrix(i, j) +=
                (time_step * theta *
                fe_iv.jump(i, qpoint) // [\phi_i]
                *
                fe_iv.shape_value((beta_dot_n > 0), j, qpoint) // phi_j^{upwind}
                * beta_dot_n                                   // (\beta . n)
                * JxW[qpoint]);                                 // dx
              
              Tensor<1, dim> e_y;
              e_y[0] = 0;
              e_y[1] = 1;
              
              copy_data_face.cell_matrix(i, j) +=
                (time_step * theta * D / cell->diameter() * 10.0 *
                std::abs(normals[qpoint] * e_y)   // check if n is in y direction
                * fe_iv.jump(i, qpoint)           // [\phi_i]
                * fe_iv.jump(j, qpoint)           // [\phi_j]
                * JxW[qpoint]);                   // dx
              
              copy_data_face.cell_matrix(i, j) +=
                (-time_step * theta * D
                * fe_iv.average_gradient(i, qpoint)[1]  // { \nabla \phi_i }[p]
                * normals[qpoint][1]                    // n[p]
                * fe_iv.jump(j, qpoint)                 // [\phi_j]
                * JxW[qpoint]);                         // dx
              
              copy_data_face.cell_matrix(i, j) +=
                (-time_step * theta * D
                * fe_iv.jump(i, qpoint)                 // [\phi_i]
                * fe_iv.average_gradient(j, qpoint)[1]  // { \nabla \phi_j }
                * normals[qpoint][1]                    // n
                * JxW[qpoint]);                         // dx
              
            }
              
        }
    };
 
    const AffineConstraints<double> constraints;
 
    const auto copier = [&](const CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             right_hand_side);
 
      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };
 
    ScratchData<dim> scratch_data(mapping, fe, quadrature, quadrature_face);
    CopyData         copy_data;
 
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

  template <int dim>
  double DoubleWell<dim>::compute_norm()
  {

    QGauss<dim> quadrature_formula(dof_handler.get_fe().degree + 1);
    FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula, update_values | update_JxW_values);

    std::vector<double> solution_values(quadrature_formula.size());
    double integral = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, solution_values);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            integral += solution_values[q_index] * fe_values.JxW(q_index);
        }
    }

    return integral;
  }
 
  template <int dim>
  void DoubleWell<dim>::solve()
  {
    SolverControl                    solver_control(1000, 1e-12);
    SolverRichardson<Vector<double>> solver(solver_control);
 
    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
 
    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());
 
    solver.solve(system_matrix, solution, right_hand_side, preconditioner);
 
    std::cout << "  Solver converged in " << solver_control.last_step()
              << " iterations." << std::endl;
  }
 
 
  template <int dim>
  void DoubleWell<dim>::refine_grid()
  {
    Vector<float> gradient_indicator(triangulation.n_active_cells());
 
    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  solution,
                                                  gradient_indicator);
 
    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);
 
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    gradient_indicator,
                                                    0.4,
                                                    0.46);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->refine_flag_set() && cell->level() > 7)
        cell->clear_refine_flag();
 
    SolutionTransfer<dim> solution_trans(dof_handler);
 
    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
 
    triangulation.execute_coarsening_and_refinement();
    setup_system();
 
    solution_trans.interpolate(previous_solution, solution);
    /*constraints.distribute(solution);*/
  }
 
 
  template <int dim>
  void DoubleWell<dim>::output_results() const
  {
    std::cout << "Output results 1" << std::endl;
    DataOut<dim> data_out;
 
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(solution, "U");

    data_out.build_patches();
 
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
 
    const std::string filename =
      "solutions/solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }
 
 
  template <int dim>
  void DoubleWell<dim>::run()
  {
    const unsigned int initial_global_refinement       = 5;
    const unsigned int n_adaptive_pre_refinement_steps = 5;
 
    GridGenerator::hyper_rectangle(triangulation, Point<dim>(-6.5, -21.0), Point<dim>(6.5, 21.0));
    triangulation.refine_global(initial_global_refinement);
 
    setup_system();
 
    unsigned int pre_refinement_step = 0;
 
    Vector<double> tmp;
    Vector<double> forcing_terms;

    start_time_iteration:
    std::cout << pre_refinement_step << std::endl;
    time            = 0.0;
    timestep_number = 0;
 
    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());
    old_solution.reinit(solution.size());
    
    VectorTools::interpolate(dof_handler,
                             initial_condition<dim>(),
                             old_solution);
    solution = old_solution;
    output_results();
    while (time <= 8.281)
      {
        time += time_step;
        ++timestep_number;
 
        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;
        setup_system();

        assemble_system();
        mass_matrix.vmult_add(right_hand_side, old_solution);

        /*constraints.condense(system_matrix, system_rhs);
 
        {
          BoundaryValues<dim> boundary_values_function;
          boundary_values_function.set_time(time);
 
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);
 
          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }*/
 
        solve();
        double norm = compute_norm();
        std::cout << "Norm: " << norm << std::endl;
 
        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_grid();
            std::cout << "  Number of active cells:       "
                    << triangulation.n_active_cells() << std::endl;
            ++pre_refinement_step;
 
            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
 
            std::cout << std::endl;
            goto start_time_iteration;
          }
        else if ((timestep_number > 0) && (timestep_number % 10 == 0))
          {
            refine_grid();
            std::cout << "  Number of active cells:       "
                    << triangulation.n_active_cells() << std::endl;
            tmp.reinit(solution.size());
            forcing_terms.reinit(solution.size());
            output_results();
          }
        old_solution = solution;
      }
  }
}
 
 
int main()
{
  try
    {
      PlanckFokker::DoubleWell<2> dgmethod;
      dgmethod.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}