<?php

/**
 * Description of meta
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\void;

class single extends void{

    
    public function __construct($tag = "meta") {
        parent::__construct();
        $this->tag = $tag;
    }


}
