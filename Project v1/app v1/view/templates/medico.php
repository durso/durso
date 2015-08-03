<?php

$html = "<div class=\"row\">"
                ."<div class=\"col-lg-12 text-center\">"
                    ."<h1>".$row["nome"]."</h1>"
                    ."<hr class=\"primary doctor\">"
                ."</div>"
            ."</div>"
            ."<div class=\"row\">"
                ."<div class=\"col-lg-4 col-lg-offset-2\">"
                    ."<p>CRM: ".$row["crm"]."</p>"
                    ."<p>Estado: ".$row["estado"]."</p>"
                ."</div>"
                ."<div class=\"col-lg-4 right\">"
                    ."<p>"
                        .stars($row["rating"])
                    ."</p>"
                    ."<p>"
                        .numAvaliacoes($row["avaliacoes"])
                    ."</p>"
                ."</div>"
            ."</div>";
return $html;