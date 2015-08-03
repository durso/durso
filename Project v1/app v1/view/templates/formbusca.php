<?php

$html = "<form action=\"/buscar/resultado\" method=\"get\">"
          ."<div class=\"form-group\">"
            ."<label for=\"nome\">Nome:</label>"
            ."<input type=\"text\" class=\"form-control\" id=\"nome\" name=\"nome\" placeholder=\"Nome do medico\">"
          ."</div>"
          ."<div class=\"form-group\">"
            ."<label for=\"crm\">CRM:</label>"
            ."<input type=\"text\" class=\"form-control\" id=\"crm\" name=\"crm\" placeholder=\"Numero do CRM\">"
          ."</div>"
          ."<div class=\"form-group\">"
            ."<label for=\"estado\">Estado:</label>"
            ."<select class=\"form-control\" name=\"estado\" id=\"estado\">"
            ."<option>Todos os Estados</option>"
            .optionList($rows)
            ."</select>"
          ."</div>"
          ."<button type=\"submit\" class=\"btn btn-default\">Submit</button>"
        ."</form>";
return $html;