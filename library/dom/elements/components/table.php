<?php

/**
 * Description of table
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class table extends paired{

    
    public function __construct() {
        parent::__construct();
        $this->tag = "table";
        $this->addClass("table");
    }


}