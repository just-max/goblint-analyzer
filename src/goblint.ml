open Goblint_lib
open GobConfig
open Goblintutil
open Maingoblint
open Prelude
open Printf

(** the main function *)
let main () =
  try
    Cilfacade.init ();
    Maingoblint.reset_stats ();
    Maingoblint.parse_arguments ();
    handle_extraspecials ();
    GoblintDir.init ();

    if get_bool "dbg.verbose" then (
      print_endline (localtime ());
      print_endline Goblintutil.command_line;
    );
    let file = lazy (Fun.protect ~finally:GoblintDir.finalize preprocess_parse_merge) in
    if get_bool "server.enabled" then (
      let file =
        if get_bool "server.reparse" then
          None
        else
          Some (Lazy.force file)
      in
      Server.start file
    )
    else (
      let file = Lazy.force file in
      let changeInfo =
        if GobConfig.get_bool "incremental.load" || GobConfig.get_bool "incremental.save" then
          diff_and_rename file
        else
          Analyses.empty_increment_data ()
      in
      file |> do_analyze changeInfo;
      do_stats ();
      do_html_output ();
      do_gobview ();
      if !verified = Some false then exit 3 (* verifier failed! *)
    )
  with
  | Exit ->
    do_stats ();
    exit 1
  | Sys.Break -> (* raised on Ctrl-C if `Sys.catch_break true` *)
    do_stats ();
    (* Printexc.print_backtrace BatInnerIO.stderr *)
    eprintf "%s\n" (MessageUtil.colorize ~fd:Unix.stderr ("{RED}Analysis was aborted by SIGINT (Ctrl-C)!"));
    exit 131 (* same exit code as without `Sys.catch_break true`, otherwise 0 *)
  | Timeout ->
    do_stats ();
    eprintf "%s\n" (MessageUtil.colorize ~fd:Unix.stderr ("{RED}Analysis was aborted because it reached the set timeout of " ^ get_string "dbg.timeout" ^ " or was signalled SIGPROF!"));
    exit 124

(* We do this since the evaluation order of top-level bindings is not defined, but we want `main` to run after all the other side-effects (e.g. registering analyses/solvers) have happened. *)
let () = Printexc.record_backtrace true; at_exit main
